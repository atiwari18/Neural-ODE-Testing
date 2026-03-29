import os
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torchdiffeq import odeint

def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      nsample=100,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      savefig=True):

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('./ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    samp_start_idxs = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample
        samp_start_idxs.append(t0_idx)

        cc = bool(npr.rand() > .5)  # uniformly select rotation
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)
    samp_start_idxs = np.array(samp_start_idxs, dtype=np.float64)

    return orig_trajs, samp_trajs, orig_ts, samp_ts, samp_start_idxs

#ODE Function
class ODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, latent_dim)
        )

        #Small Init keep early training stable
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, t, z):
        return self.net(z)
    
#GRU Update Unit
class GRUUpdate(nn.Module):
    """
    GRU from the paper.
    Takes (h_mean, h_std, x_obs) and returns new (h_mean, h_std).
    x_obs already has the mask concatenated: shape [..., 2*input_dim].
    """

    def __init__(self, latent_dim, input_dim, n_units=100):
        super().__init__()

        d = latent_dim * 2 + input_dim

        self.update_gate = nn.Sequential(
            nn.Linear(d, n_units), 
            nn.Tanh(),
            nn.Linear(n_units, latent_dim), 
            nn.Sigmoid()
        )

        self.reset_gate = nn.Sequential(
            nn.Linear(d, n_units), 
            nn.Tanh(), 
            nn.Linear(n_units, latent_dim), 
            nn.Sigmoid()
        )

        self.new_state_net = nn.Sequential(
            nn.Linear(d, n_units), 
            nn.Tanh(), 
            nn.Linear(n_units, latent_dim * 2)
        )

        for net in [self.update_gate, self.reset_gate, self.new_state_net]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, h_mean, h_std, x):
        cat = torch.cat([h_mean, h_std, x], dim=-1)
        u = self.update_gate(cat)
        r = self.reset_gate(cat)
        cat2 = torch.cat([h_mean * r, h_std * r, x], dim=-1)

        out = self.new_state_net(cat2)
        new_h, new_s = out.chunk(2, dim=-1)
        new_s = new_s.abs()

        new_h = (1 - u) * new_h + u * h_mean
        new_s = (1 - u) * new_s + u * h_std

        #masked update: only update hidden state where at least one feature present
        n_data = x.size(-1) // 2
        mask = x[..., n_data:]
        any_obs = (mask.sum(-1, keepdim=True) > 0).float()
        new_h = any_obs * new_h + (1 - any_obs) * h_mean
        new_s = any_obs * new_s + (1 - any_obs) * h_std
        new_s = new_s.abs()

        return new_h, new_s
    
class ODERNNEncoder(nn.Module):
    """
    Runs backwards through the observation sequence.
    At each step:
      1. Evolve h from prev_t to t_i via the ODE solver (backwards in time).
      2. Update h with the GRU given the observation at t_i.
    Finally maps the last h to (mu_z0, sigma_z0).
    """
 
    def __init__(self, latent_dim, input_dim, z0_dim, ode_hidden=64, gru_units=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.z0_dim     = z0_dim
 
        # ODE for the encoder hidden state
        self.ode_func = ODEFunc(latent_dim, hidden_dim=ode_hidden)
 
        # GRU update (input includes mask, so input_dim*2)
        self.gru = GRUUpdate(latent_dim, input_dim * 2, n_units=gru_units)
 
        # Project final hidden state to z0 distribution parameters
        self.to_z0 = nn.Sequential(
            nn.Linear(latent_dim * 2, 100),
            nn.Tanh(),
            nn.Linear(100, z0_dim * 2),
        )
        for m in self.to_z0.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
 
    def forward(self, data_w_mask, time_steps):
        """
        data_w_mask: (n_traj, n_tp, 2*input_dim)
        time_steps:  (n_tp,)
        Returns: mu_z0 (1, n_traj, z0_dim), sigma_z0 (1, n_traj, z0_dim)
        """
        n_traj, n_tp, _ = data_w_mask.shape
        device = data_w_mask.device
 
        h_mean = torch.zeros(1, n_traj, self.latent_dim, device=device)
        h_std  = torch.zeros(1, n_traj, self.latent_dim, device=device)
 
        # Process observations from last to first (backwards)
        for i in reversed(range(n_tp)):
            t_i    = time_steps[i]
            obs_i  = data_w_mask[:, i, :].unsqueeze(0)   # (1, n_traj, 2*input_dim)
 
            # ODE step: if not at the very last time point, evolve h backwards
            if i < n_tp - 1:
                t_prev = time_steps[i + 1]
                # integrate from t_prev → t_i  (t_i < t_prev, so going backwards)
                t_span = torch.stack([t_prev, t_i])
                h_mean = odeint(self.ode_func, h_mean, t_span,
                                rtol=1e-3, atol=1e-4, method="euler")[-1]
 
            # GRU update with observation
            h_mean, h_std = self.gru(h_mean, h_std, obs_i)
 
        # Map to z0 distribution
        h_cat = torch.cat([h_mean, h_std], dim=-1)  # (1, n_traj, 2*latent_dim)
        z0_params = self.to_z0(h_cat)
        mu, log_sigma = z0_params.chunk(2, dim=-1)
        sigma = log_sigma.abs()
 
        return mu, sigma
    
class Decoder(nn.Module):
    """
    Maps latent z(t) back to observation space x(t).
    Deliberately shallow — all dynamics live in the ODE.
    Shape: (*, latent_dim) -> (*, input_dim)
    """
 
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.net = nn.Linear(latent_dim, input_dim)
        nn.init.orthogonal_(self.net.weight)
        nn.init.zeros_(self.net.bias)
 
    def forward(self, z):
        return self.net(z)
    
class LatentODE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        rec_dim,
        ode_hidden=64,
        gru_units=100,
        obsrv_std=0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.obsrv_std  = torch.tensor([obsrv_std])
 
        self.encoder  = ODERNNEncoder(rec_dim, input_dim, latent_dim,
                                       ode_hidden=ode_hidden, gru_units=gru_units)
        self.ode_func = ODEFunc(latent_dim, hidden_dim=ode_hidden)
        self.decoder  = Decoder(latent_dim, input_dim)
 
        # Standard Gaussian prior on z0
        self.z0_prior = Normal(
            torch.zeros(latent_dim),
            torch.ones(latent_dim),
        )
 
    def get_reconstruction(self, observed_data, observed_ts, pred_ts,
                            mask=None, n_samples=1):
        """
        observed_data: (n_traj, n_obs, input_dim)
        observed_ts:   (n_obs,)
        pred_ts:       (n_pred,)
        mask:          (n_traj, n_obs, input_dim) or None
 
        Returns:
          pred_x:      (n_samples, n_traj, n_pred, input_dim)
          info dict with 'first_point' and 'kl_z0'
        """
        device = observed_data.device
        self.obsrv_std = self.obsrv_std.to(device)
 
        #Build data+mask input for encoder
        if mask is None:
            mask = torch.ones_like(observed_data)
        data_w_mask = torch.cat([observed_data, mask], dim=-1)   # (n_traj, n_obs, 2*dim)
 
        #Encode → q(z0|x)
        mu_z0, sigma_z0 = self.encoder(data_w_mask, observed_ts)   # (1, n_traj, latent_dim)
 
        #Sample z0 (reparameterisation trick), replicate for n_samples
        mu_rep    = mu_z0.expand(n_samples, -1, -1)
        sigma_rep = sigma_z0.expand(n_samples, -1, -1)
        eps       = torch.randn_like(mu_rep)
        z0_sample = mu_rep + sigma_rep * eps                        # (n_samples, n_traj, latent_dim)
 
        #Solve ODE forward in time
        #odeint expects (n_samples*n_traj, latent_dim) if we batch
        n_samp, n_traj, d = z0_sample.shape
        z0_flat   = z0_sample.reshape(n_samp * n_traj, d)
 
        sol = odeint(self.ode_func, z0_flat, pred_ts,
                     rtol=1e-3, atol=1e-4, method="dopri5")
        #sol: (n_pred, n_samp*n_traj, latent_dim)
        sol = sol.permute(1, 0, 2)                                  # (n_samp*n_traj, n_pred, d)
        sol = sol.reshape(n_samp, n_traj, len(pred_ts), d)          # (n_samp, n_traj, n_pred, d)
 
        #Decode
        pred_x = self.decoder(sol)                                  # (n_samp, n_traj, n_pred, input_dim)
 
        info = {"first_point": (mu_z0, sigma_z0, z0_sample)}
        return pred_x, info
 
    def compute_loss(self, batch_data, batch_ts, kl_coef=1.0, n_samples=1):
        """
        batch_data: (n_traj, n_tp, input_dim) — observed noisy trajectory
        batch_ts:   (n_tp,)
        We use the full sequence as both observed and predicted.
        """
        pred_x, info = self.get_reconstruction(
            batch_data, batch_ts, batch_ts, n_samples=n_samples
        )
        mu_z0, sigma_z0, _ = info["first_point"]
        sigma_z0 = sigma_z0.abs()
 
        #Reconstruction likelihood (Gaussian)
        #pred_x: (n_samp, n_traj, n_tp, input_dim)
        #truth:  (n_traj, n_tp, input_dim)
        truth = batch_data.unsqueeze(0).expand_as(pred_x)
        std   = self.obsrv_std.expand_as(pred_x)
        log_p = Normal(pred_x, std).log_prob(truth)        # (n_samp, n_traj, n_tp, d)
        rec_ll = log_p.mean(dim=(-1, -2))                  # (n_samp, n_traj) — mean over tp & dim
 
        #KL divergence  KL( q(z0|x) || N(0,I) )
        q_z0 = Normal(mu_z0, sigma_z0)                     # (1, n_traj, latent_dim)
        p_z0 = Normal(
            torch.zeros_like(mu_z0),
            torch.ones_like(sigma_z0),
        )
        kl = kl_divergence(q_z0, p_z0).mean(dim=-1)       # (1, n_traj)
        kl = kl.expand(n_samples, -1)                      # (n_samp, n_traj)
 
        #ELBO (IWAE bound for n_samples > 1) 
        elbo = rec_ll - kl_coef * kl                       # (n_samp, n_traj)
        if n_samples > 1:
            loss = -torch.logsumexp(elbo, dim=0).mean()
        else:
            loss = -elbo.mean()
 
        return loss, rec_ll.mean().detach(), kl.mean().detach()