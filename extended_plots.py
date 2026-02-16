from models.neural_ode import ODEFunc, extrapolate, plot_sine_extrapolation
from models.lstm import LSTM, plot_lstm_sine_extrapolation
from dataset.data import SineDynamics, generate_sine, generate_spiral
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Ode Model Plotting")
    parser.add_argument(
        "--sine", 
        action="store_true", 
        help="Generate sine extrapolation for future vals [12π, 24π, 48π]"
    )

    return parser.parse_args()

def generate_sines(node, future_vals, t, single_true, device, true_func, lstm, seed):

    for v in future_vals:
        t_future, state_future = extrapolate(node, t, single_true[:, 0, :], device=device, t_max=v)
        plot_sine_extrapolation(t, single_true[:, 0, :], t_future, state_future, true_func=true_func, file_name=f"sine_extrapolation ({v}).png", model=node, device=device)

    #loop for lstm
    for v in future_vals:
        lstm_all, t_all = lstm.rollout(seed, t_train=t, t_max=v, device=device)
        lstm_all = lstm_all[:, 0, :]
        plot_lstm_sine_extrapolation(t_train=t, state_train=single_true[:, 0, :], 
                             t_all=t_all, lstm_all=lstm_all, 
                             true_func=true_func, t_max=v, file_name="lstm_sine_extrapolation ({v}).png", device=device)

    print(f"Generated all plots for future_vals: [12π, 24π, 48π]")

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
        weights = torch.load(".\\Results\\neural_ode_sine.pth", weights_only=True)
        node.load_state_dict(weights)

        #Load LSTM
        lstm = LSTM(input_dim=2, hidden_dim=64, num_layers=2, output_dim=2).to(device)
        lstm_weights = torch.load(".\\Results\\lstm_sine.pth", weights_only=True)
        lstm.load_state_dict(lstm_weights)

        #single values
        single_true = true_traj[:, 0:1, :]

        generate_sines(node, future_vals, t, single_true, device, true_func)

        #LSTM
        seed = true_traj[:, 0, :]
       

