import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
import os
from models.neural_ode import ODEFunc
from dataset.data import SpiralDynamics, generate_spiral

#Encoder
#This is a backwards RNN - it process the observations in reverse order to product an initial
#latent state that has information about the trajectory.
class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, hidden_dim=25):
        super(RecognitionRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        #Input to Hidden: Combines observation + previous hidden state
        self.i2h = nn.Linear(obs_dim + hidden_dim, hidden_dim)

        #Hidden to Output --> produces latent mean and log-variance
        self.h2o = nn.Linear(hidden_dim, latent_dim*2)

    def forward(self, x, h):
        #Concatenate observation with the hidden state 
        combine = torch.cat((x, h), dim=1)

        #Update the hidden state with the tanh activation
        h = torch.tanh(self.i2h(combine))
        
        #Produce the latent parameters
        out = self.h2o(h)

        return out, h
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim).to(device)
    
#Decoder (latent --> observation)
class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, hidden_dim=20):
        super(Decoder, self).__init__()

        #MLP to decode from latent space to data space
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, obs_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, z):
        #handle both 3d and 2d inputs
        original_shape = z.shape
        if len(z.shape) == 3:
            #reshape [time, batch, latent] --> [time*batch, latent]
            z = z.reshape(-1, z.shape[-1])

        #decode!!!
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)

        #Reshape back if needed
        if len(original_shape) == 3:
            out = out.reshape(original_shape[0], original_shape[1], -1)

        return out

class LatentODE(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, encoder_hidden=25, ode_hidden=20, decoder_hidden=20):
        super(LatentODE, self).__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        #3 network components
        self.encoder = RecognitionRNN(latent_dim, obs_dim, encoder_hidden)
        self.ode_func = ODEFunc(latent_dim, ode_hidden, time_invariant=True)
        self.decoder = Decoder(latent_dim, obs_dim, decoder_hidden)

    def encode(self, observed_data, observed_times):
        """
        Enocder observation sequence into initial latent state z0.

        Processes the data BACKWARDS (from last to dirst) to get z0
        """
        batch_size = observed_data.shape[0]
        seq_len = observed_data.shape[1]

        #initialize hidden state
        h = self.encoder.init_hidden(batch_size, observed_data.device)

        #Process observation backwards through time, this allows the RNN to
        #accumulate information from future to past
        for t in reversed(range(seq_len)):
            obs = observed_data[:, t, :]
            out, h = self.encoder(obs, h)

        #split output into mean and log-variance
        z0_mean = out[:, :self.latent_dim]
        z0_logvar = out[:, self.latent_dim:]

        return z0_mean, z0_logvar
    
    def reparametrize(self, mean, logvar):
        #compute the standard deviation from logvar
        std = torch.exp(0.5 * logvar)

        #sample epsilon from standard normal
        eps = torch.randn_like(std)

        #Reparametrize = μ + σ * ε
        return mean + eps * std
    
    def forward(self, observed_data, observed_times, prediction_times):
        #Encode observations to get initial latent state
        z0_mean, z0_var = self.encode(observed_data, observed_times)

        #Sample latent using reparametrization trick
        z0 = self.reparametrize(z0_mean, z0_var)

        #Solve ODE in latent space
        z_traj = odeint(self.ode_func, z0, prediction_times, method="dopri5")

        #Decode latent trajectory to observations
        predicted_obs = self.decoder(z_traj)

        return predicted_obs, z0_mean, z0_var
    
def latent_ode_loss(predicted, target, z0_mean, z0_log_var, kl_weight=1.0):
    #Reconstruction loss
    recon_loss = torch.mean((predicted-target) ** 2)

    #KL Divergence = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + z0_log_var - z0_mean.pow(2) - z0_log_var.exp())

    #total loss 
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss

def train_latent_ode(model, trajs, t, epochs=200, lr=0.001, kl_weight_start=0, kl_weight_end=0.01, device="cuda", file_name="latent_ode.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #rearrange trajectories to be [batch, n, obs_dim]
    traj_batch = trajs.permute(1, 0, 2)
    
    losses = {"total": [], "recon": [], "kl": []}

    model.train()
    for epoch in range(epochs):
        #KL Annealing
        kl_weight = kl_weight_start + (kl_weight_end - kl_weight_start) * min(epoch / (epochs * 0.5), 1.0)

        optimizer.zero_grad()

        #forward pas
        predicted, z0_mean, z0_logvar = model(traj_batch, t, t)

        #compute loss
        total_loss, recon_loss, kl_loss = latent_ode_loss(predicted, trajs, z0_mean, z0_logvar, kl_weight)

        #backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        #Record losses 
        losses["total"].append(total_loss.item())
        losses["recon"].append(recon_loss.item())
        losses["kl"].append(kl_loss.item())

        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Total: {total_loss.item():.6f} | "
                  f"Recon: {recon_loss.item():.6f} | "
                  f"KL: {kl_loss.item():.6f} | "
                  f"KL Weight: {kl_weight:.6f}")
            
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    full_path = os.path.join(results_dir, file_name)

    torch.save(model.state_dict(), full_path)

    print(f"Model saved to {full_path}")
            
    return losses

def plot_loss(losses, file_name="Latent ODE Losses.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses["total"], label='Training Loss')
    plt.title('Latent ODE Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)    
    full_path = os.path.join(results_dir, file_name)

    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")

def extrapolate_latent_ode(model, observed_data, observed_times, t_max, device):
    """
    Extrapolate the trajectory using a latent ode
    """

    model.eval()

    with torch.no_grad():
        z0_mean, z0_logvar = model.encode(observed_data, observed_times)

        #for extrapolation we need the mean
        z0 = z0_mean

        #timepoints 
        dt = (observed_times[1] - observed_times[0]).item()
        t_start = observed_times[0].item()
        n = int( (t_max - t_start) / dt )

        t_full = torch.linspace(t_start, t_max, n).to(device)

        z_traj = odeint(model.ode_func, z0, t_full, method="dopri5")

        predicted_obs = model.decoder(z_traj)

    return t_full, predicted_obs, z_traj

def plot_latent_ode_extrapolation(t_train, state_train, t_full, predicted_full, true_func=None, 
                                   t_max=None,
                                   file_name="latent_ode_extrapolation.png",
                                   device='cpu'):

    t_train_np = t_train.cpu().numpy()
    y_train = state_train[:, 0].cpu().numpy()  # Position only
    
    t_full_np = t_full.cpu().numpy()
    y_predicted = predicted_full[:, 0].cpu().numpy()  # Position only
    
    plt.figure(figsize=(14, 6))
    
    # Ground truth
    if true_func is not None:
        gt_end = t_max if t_max is not None else t_full[-1].item()
        y0_train = state_train[0:1, :].to(device)
        
        with torch.no_grad():
            t_gt = torch.linspace(t_train[0].item(), gt_end, 500).to(device)
            state_gt = odeint(true_func, y0_train, t_gt)
            plt.plot(t_gt.cpu().numpy(), state_gt[:, 0, 0].cpu().numpy(),
                    'gray', linestyle='--', alpha=0.5, linewidth=2.5,
                    label='True Dynamics')
    
    # Latent ODE prediction (full trajectory)
    plt.plot(t_full_np, y_predicted, 'green', linewidth=2.5, alpha=0.8,
            label='Latent ODE Trajectory')
    
    # Training observations
    plt.scatter(t_train_np, y_train, c='red', s=40, alpha=0.7,
               zorder=5, label='Training Observations')
    
    # Mark boundary
    plt.axvline(x=t_train_np[-1], color='orange', linestyle=':',
               linewidth=2, alpha=0.7, label='End of Training')
    
    plt.title("Latent ODE: Sine Wave Extrapolation", fontsize=14, fontweight='bold')
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Position (y)", fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, file_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {os.path.join(results_dir, file_name)}")

def plot_spiral_extrapolation(t_train, state_train, t_full, predicted_full, 
                              true_traj=None,
                              file_name="latent_ode_spiral_extrapolation.png",
                              device='cpu'):
    # Convert to numpy
    state_train_np = state_train.cpu().numpy()
    predicted_full_np = predicted_full.cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    
    # Plot true full trajectory if available (for reference)
    if true_traj is not None:
        true_np = true_traj.cpu().numpy()
        plt.plot(true_np[:, 0], true_np[:, 1], 'gray', linestyle='--', 
                alpha=0.4, linewidth=2, label='True Training Trajectory')
    
    # Plot predicted extrapolation
    plt.plot(predicted_full_np[:, 0], predicted_full_np[:, 1], 
            'green', linewidth=2.5, alpha=0.8, label='Latent ODE Extrapolation')
    
    # Plot training observations
    plt.scatter(state_train_np[:, 0], state_train_np[:, 1], 
               c='red', s=40, alpha=0.7, zorder=5, label='Training Observations')
    
    # Mark start and end
    plt.scatter([state_train_np[0, 0]], [state_train_np[0, 1]], 
               c='green', s=150, marker='o', edgecolors='black', 
               linewidth=2, zorder=10, label='Start')
    plt.scatter([predicted_full_np[-1, 0]], [predicted_full_np[-1, 1]], 
               c='blue', s=150, marker='x', linewidth=3, 
               zorder=10, label='Extrapolation End')
    
    plt.title("Latent ODE: Spiral Extrapolation (Phase Space)", 
             fontsize=14, fontweight='bold')
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, file_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {os.path.join(results_dir, file_name)}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Generate data
    true_func = SpiralDynamics(device=device).to(device)
    t, y0, trajs = generate_spiral(batch_size=16, n_samples=100, t_max=4*torch.pi, device=device)
    t = t.to(device)
    y0 = y0.to(device)
    trajs = trajs.to(device)

    #Create Model
    model = LatentODE(latent_dim=4,
                      obs_dim=2, 
                      encoder_hidden=25, 
                      ode_hidden=64,
                      decoder_hidden=25).to(device)
    
    #Train
    print("\nTraining Latent ODE...")
    losses = train_latent_ode(model, trajs, t, epochs=1000, kl_weight_start=0, kl_weight_end=0.001, device=device, file_name="latent_ode_spiral.pth")
    plot_loss(losses, file_name="Latent ODE Losses (spiral).png")





