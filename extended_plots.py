from models.neural_ode import AugmentedNODEFunc, ODEFunc, extrapolate, plot_sine_extrapolation, plot_learned_dynamics_vs_true, plot_comparison
from models.lstm import LSTM, plot_lstm_sine_extrapolation
from dataset.data import SineDynamics, generate_sine, generate_spiral
from torchdiffeq import odeint_adjoint as odeint
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

    parser.add_argument("--node", action="store_true")
    parser.add_argument("--lstm", action="store_true")
    parser.add_argument("--node_dynamics", action="store_true")
    parser.add_argument("--plot_comparison", action="store_true")
    parser.add_argument("--plot_clean", action="store_true")

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


if __name__ == '__main__':
    args = parse_args()

    future_vals = [12 * torch.pi, 24 * torch.pi, 48 * torch.pi]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Load Data
    print("\nLoading Data...")
    true_func = SineDynamics(device=device).to(device)
    t, y0, true_traj = generate_sine(true_func, batch_size=16, n_samples=100, t_max=4*torch.pi, device=device)
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
       

