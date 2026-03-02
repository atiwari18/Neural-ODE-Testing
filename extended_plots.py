from models.neural_ode import AugmentedNODEFunc, ODEFunc, extrapolate, plot_sine_extrapolation, plot_learned_dynamics_vs_true, plot_comparison
from models.lstm import LSTM, plot_lstm_sine_extrapolation
from models.latent_ode import LatentODE, extrapolate_latent_ode, plot_latent_ode_extrapolation, plot_spiral_extrapolation
from dataset.data import SineDynamics, generate_sine, generate_spiral, SpiralDynamics
from torchdiffeq import odeint_adjoint as odeint
import torch
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("Ode Model Plotting")
    parser.add_argument(
        "--sine", 
        action="store_true", 
        help="Generate sine extrapolation for future vals [12π, 24π, 48π]"
    )

    parser.add_argument("--node", action="store_true")
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--node_dynamics", action="store_true")
    parser.add_argument("--plot_comparison", action="store_true")
    parser.add_argument("--plot_clean", action="store_true")
    parser.add_argument("--latent_ode", action="store_true")
    parser.add_argument("--spiral", action="store_true")

    return parser.parse_args()

def plot_nfe_comparison(labels, nfes, file_name="nfe_by_horizon_node (sine-1000).png"):
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, nfes, color='steelblue', edgecolor='black', width=0.5)
    
    # Annotate bars with exact counts
    for bar, nfe in zip(bars, nfes):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(nfe), ha='center', va='bottom', fontsize=11)
    
    plt.title("Neural ODE: Function Evaluations by Extrapolation Horizon", fontsize=13)
    plt.xlabel("Extrapolation Horizon")
    plt.ylabel("Number of Function Evaluations (NFE)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    full_path = os.path.join(results_dir, file_name)

    plt.savefig(full_path)

    print(f"NFE plot saved to: {full_path}")

def generate_sines(ode_model, future_vals, t, single_true, device, true_func, lstm, seed, is_node=False, is_lstm=True):
    node_nfes = []
    labels = []

    if is_node:
        for v in future_vals:
            t_future, state_future, nfe = extrapolate(node, t, single_true[:, 0, :], device=device, t_max=v)
            node_nfes.append(nfe)
            labels.append(f"{v/torch.pi:.0f}π")
            plot_sine_extrapolation(t, single_true[:, 0, :], t_future, state_future, true_func=true_func, file_name=f"anode_sine_500-3-reg ({v/torch.pi:.0f}π).png", model=ode_model, device=device)
            
        plot_nfe_comparison(labels, node_nfes, file_name="anode_sine_500-3-reg-nfes.png")

    #loop for lstm
    if is_lstm:
        for v in future_vals:
            lstm_all, t_all = lstm.rollout(seed, t_train=t, t_max=v, device=device)
            lstm_all = lstm_all[:, 0, :]
            plot_lstm_sine_extrapolation(t_train=t, state_train=single_true[:, 0, :], 
                                t_all=t_all, lstm_all=lstm_all, 
                                true_func=true_func, t_max=v, file_name=f"lstm_sine_extrapolation (test-{v}).png", device=device)
        

    print(f"Generated all plots for future_vals: [{future_vals[0]/torch.pi:.0f}π, {future_vals[1]/torch.pi:.0f}π, {future_vals[2]/torch.pi:.0f}π]")

    return

@torch.no_grad()
def inspect_encoder_latents(model, observed_data, observed_times, 
                           n_samples=8, return_df=False, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    observed_data = observed_data.to(device)
    observed_times = observed_times.to(device)
    
    if observed_data.dim() == 2:
        observed_data = observed_data.unsqueeze(0)   # add batch dim if needed
    
    batch_size = observed_data.shape[0]
    idx = np.random.choice(batch_size, size=min(n_samples, batch_size), replace=False)
    
    print(f"\nInspecting encoder outputs for {len(idx)} random trajectories:\n")
    print("   idx   |  z₀ mean                  |  z₀ std                   | sample z₀")
    print("-" * 85)
    
    rows = []
    
    for i, sample_idx in enumerate(idx):
        x = observed_data[sample_idx:sample_idx+1]      # [1, T, obs_dim]
        t = observed_times                               # [T] or [1,T]
        
        z0_mean, z0_logvar = model.encode(x, t)
        z0_std = torch.exp(0.5 * z0_logvar)
        
        # one sample from posterior
        z0_sample = model.reparametrize(z0_mean, z0_logvar)
        
        mean_str = " ".join(f"{v:.4f}" for v in z0_mean[0].cpu().numpy())
        std_str  = " ".join(f"{v:.4f}" for v in z0_std[0].cpu().numpy())
        samp_str = " ".join(f"{v:.4f}" for v in z0_sample[0].cpu().numpy())
        
        print(f"{sample_idx:6d}   | {mean_str} | {std_str} | {samp_str}")

def visualize_latent_ode(model, samp_trajs, samp_ts, orig_trajs, orig_ts, device, 
                          file_name="latent_ode_vis.png"):
    model.eval()
    with torch.no_grad():
        # Encode all sampled trajectories to get z0
        traj_batch = samp_trajs.permute(1, 0, 2)  # [batch, time, dim] -> encode expects this
        z0_mean, z0_logvar = model.encode(traj_batch, samp_ts)
        
        # Sample z0 using reparametrization
        eps = torch.randn(z0_mean.size()).to(device)
        z0 = eps * torch.exp(0.5 * z0_logvar) + z0_mean
        
        # Take first trajectory only
        z0 = z0[0]  # [latent_dim]

        # Positive and negative time directions (author's approach)
        ts_pos = np.linspace(0., 2. * np.pi, num=2000)
        ts_neg = np.linspace(-np.pi, 0., num=2000)[::-1].copy()
        ts_pos = torch.from_numpy(ts_pos).float().to(device)
        ts_neg = torch.from_numpy(ts_neg).float().to(device)

        # Integrate ODE in both directions
        zs_pos = odeint(model.ode_func, z0, ts_pos)
        zs_neg = odeint(model.ode_func, z0, ts_neg)

        # Decode
        xs_pos = model.decoder(zs_pos)
        xs_neg = torch.flip(model.decoder(zs_neg), dims=[0])

    # Convert to numpy
    xs_pos = xs_pos.cpu().numpy()
    xs_neg = xs_neg.cpu().numpy()
    orig_traj = orig_trajs[0].cpu().numpy()  # expects [batch, time, 2]
    samp_traj = samp_trajs[0].cpu().numpy()  # expects [batch, time, 2]

    plt.figure()
    plt.plot(orig_traj[:, 0], orig_traj[:, 1], 'g', label='true trajectory')
    plt.plot(xs_pos[:, 0], xs_pos[:, 1], 'r', label='learned trajectory (t>0)')
    plt.plot(xs_neg[:, 0], xs_neg[:, 1], 'c', label='learned trajectory (t<0)')
    plt.scatter(samp_traj[:, 0], samp_traj[:, 1], label='sampled data', s=3)
    plt.legend()

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)

    full_path = os.path.join(results_dir, file_name)
    plt.savefig(full_path, dpi=500)
    plt.close()
    print(f"Saved visualization to {full_path}")


if __name__ == '__main__':
    args = parse_args()

    future_vals = [12 * torch.pi, 24 * torch.pi, 48 * torch.pi]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Load Data
    print("\nLoading Data...")
    if args.sine:
        true_func = SineDynamics(device=device).to(device)
        t, y0, true_traj = generate_sine(true_func, batch_size=16, n_samples=100, t_max=4*torch.pi, device=device)
    elif args.spiral:
        true_func = SpiralDynamics(device=device).to(device)
        t, y0, true_traj = generate_spiral(batch_size=16, n_samples=100, t_max=4*torch.pi, device=device)

    t = t.to(device)
    y0 = y0.to(device)
    true_traj = true_traj.to(device)
    print("Data Loaded!\n")

    #load Neural ODE
    node = ODEFunc(time_invariant=True).to(device)
    anode = AugmentedNODEFunc(time_invariant=True, augment_dim=1).to(device)
    weights = torch.load(".\\Results\\anode_sine_500-3-reg-0.5.pth", weights_only=True)
    anode.load_state_dict(weights)

    #Load LSTM
    lstm = LSTM(input_dim=2, hidden_dim=64, num_layers=2, output_dim=2).to(device)
    lstm_weights = torch.load(".\\Results\\lstm_sine.pth", weights_only=True)
    lstm.load_state_dict(lstm_weights)

    #Load Latent ODE
    latent_ode = LatentODE(latent_dim=4, obs_dim=2, encoder_hidden=25, ode_hidden=64, decoder_hidden=25).to(device)
    latent_ode_weights = torch.load(".\\Results\\latent_ode_spiral-5-paper.pth", weights_only=True)
    latent_ode.load_state_dict(latent_ode_weights)

    if args.sine:
        #single values
        single_true = true_traj[:, 0:1, :]

        #LSTM
        single = true_traj[:, 0, :]
        seed = single[:20].unsqueeze(0).to(device)

        #Existence Bools
        is_node = args.node
        is_lstm = args.lstm

        generate_sines(anode, future_vals, t, single_true, device, true_func, lstm, seed, is_node=is_node, is_lstm=is_lstm)

    elif args.node_dynamics:
        plot_learned_dynamics_vs_true(node, device, file_name="learned vs. true (sine-500).png")

    elif args.plot_comparison:
        plot_comparison(true_func, anode, device, y0=y0, file_name="anode_comparison_sine.png")

    elif args.plot_clean:
        clean_y0 = torch.tensor([[-1.25, 0.0]], device=device)  # position=1, velocity=0
        clean_true = odeint(true_func, clean_y0, t)            # shape: [n_samples, 1, 2]
        t_future, state_future, nfe = extrapolate(
            anode, t, clean_true[:, 0, :], device=device, t_max=48*torch.pi
        )
        

        plot_sine_extrapolation(
            t, clean_true[:, 0, :], t_future, state_future,
            true_func=true_func, 
            file_name="clean-test 48π.png", 
            model=anode, device=device
        )

    elif args.latent_ode:
        from dataset.data import generate_spiral2d

        # Load the same data used during training
        orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
            nspiral=1000,
            ntotal=500,
            nsample=100,
            start=0.,
            stop=6 * np.pi,
            noise_std=0.3,
            a=0.,
            b=0.3,
            savefig=False
        )

        orig_trajs_tensor = torch.from_numpy(orig_trajs).float().to(device)  # [1000, 500, 2]
        samp_trajs_tensor = torch.from_numpy(samp_trajs).float().to(device)  # [1000, 100, 2]
        samp_ts_tensor = torch.from_numpy(samp_ts).float().to(device)
        orig_ts_tensor = torch.from_numpy(orig_ts).float().to(device)

        visualize_latent_ode(
            model=latent_ode,
            samp_trajs=samp_trajs_tensor,
            samp_ts=samp_ts_tensor,
            orig_trajs=orig_trajs_tensor,
            orig_ts=orig_ts_tensor,
            device=device,
            file_name="latent_ode_vis (spiral-5-paper).png"
        )
        
        latent_ode.eval()
        with torch.no_grad():
            traj_batch = true_traj.permute(1, 0, 2)
            predicted, _, _ = latent_ode(traj_batch, t, t)
            
            # Check reconstruction error
            recon_error = torch.mean((predicted - true_traj) ** 2)
            print(f"Reconstruction error: {recon_error.item():.6f}")

        traj_batch = true_traj.permute(1, 0, 2)

        inspect_encoder_latents(latent_ode, traj_batch, t, n_samples=10)

       

