from models.neural_ode import AugmentedNODEFunc, ODEFunc, extrapolate, plot_sine_extrapolation, plot_learned_dynamics_vs_true, plot_comparison
from models.lstm import LSTM, plot_lstm_sine_extrapolation
from dataset.data import SineDynamics, generate_sine, generate_spiral
import torch
import matplotlib.pyplot as plt
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Ode Model Plotting")
    parser.add_argument(
        "--sine", 
        action="store_true", 
        help="Generate sine extrapolation for future vals [12π, 24π, 48π]"
    )
    parser.add_argument("--node_dynamics", action="store_true")
    parser.add_argument("--plot_comparison", action="store_true")

    return parser.parse_args()

def plot_nfe_comparison(labels, nfes, file_name="nfe_by_horizon_anode.png"):
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

def generate_sines(node, future_vals, t, single_true, device, true_func, lstm, seed):
    node_nfes = []
    labels = []

    for v in future_vals:
        t_future, state_future, nfe = extrapolate(node, t, single_true[:, 0, :], device=device, t_max=v)
        node_nfes.append(nfe)
        labels.append(f"{v/torch.pi:.0f}π")
        plot_sine_extrapolation(t, single_true[:, 0, :], t_future, state_future, true_func=true_func, file_name=f"anode_extrapolation_250 (nfes-{v/torch.pi:.0f}π).png", model=node, device=device)

    #loop for lstm
    # for v in future_vals:
    #     lstm_all, t_all = lstm.rollout(seed, t_train=t, t_max=v, device=device)
    #     lstm_all = lstm_all[:, 0, :]
    #     plot_lstm_sine_extrapolation(t_train=t, state_train=single_true[:, 0, :], 
    #                          t_all=t_all, lstm_all=lstm_all, 
    #                          true_func=true_func, t_max=v, file_name=f"lstm_sine_extrapolation (test-{v}).png", device=device)
        
    plot_nfe_comparison(labels, node_nfes)

    print(f"Generated all plots for future_vals: [{future_vals[0]/torch.pi:.0f}π, {future_vals[1]/torch.pi:.0f}π, {future_vals[2]/torch.pi:.0f}π]")

    return


if __name__ == '__main__':
    args = parse_args()

    future_vals = [12 * torch.pi, 24 * torch.pi, 48 * torch.pi]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.sine:
        #Load Data
        print("Loading Data...")
        true_func = SineDynamics(device=device).to(device)
        t, y0, true_traj = generate_sine(true_func, batch_size=16, n_samples=100, t_max=4*torch.pi, device=device)
        t = t.to(device)
        y0 = y0.to(device)
        true_traj = true_traj.to(device)
        print("Data Loaded!\n")

        #load Neural ODE
        node = ODEFunc(time_invariant=True).to(device)
        anode = AugmentedNODEFunc(time_invariant=True, augment_dim=2).to(device)
        weights = torch.load(".\\Results\\anode_sine_250.pth", weights_only=True)
        anode.load_state_dict(weights)

        #Load LSTM
        lstm = LSTM(input_dim=2, hidden_dim=64, num_layers=2, output_dim=2).to(device)
        lstm_weights = torch.load(".\\Results\\lstm_sine.pth", weights_only=True)
        lstm.load_state_dict(lstm_weights)

        #single values
        single_true = true_traj[:, 0:1, :]

        #LSTM
        single = true_traj[:, 0, :]
        seed = single[:20].unsqueeze(0).to(device)

        generate_sines(node, future_vals, t, single_true, device, true_func, lstm, seed)

    elif args.node_dynamics:
        #load Neural ODE
        node = ODEFunc(time_invariant=True).to(device)
        weights = torch.load(".\\Results\\neural_ode_sine_500.pth", weights_only=True)
        node.load_state_dict(weights)

        plot_learned_dynamics_vs_true(node, device, file_name="learned vs. true (sine-500).png")

    elif args.plot_comparison:
        #load Neural ode
        true_func = SineDynamics(device=device).to(device)
        t, y0, true_traj = generate_sine(true_func, batch_size=16, n_samples=100, t_max=4*torch.pi, device=device)
        node = ODEFunc(time_invariant=True).to(device)
        weights = torch.load(".\\Results\\neural_ode_sine_1000.pth", weights_only=True)
        node.load_state_dict(weights)

        plot_comparison(true_func, node, device, y0=y0, file_name="comparison_sine_1000.png")
       

