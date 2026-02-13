from models.neural_ode import ODEFunc, extrapolate, plot_sine_extrapolation
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

def generate_sines(model, future_vals, t, single_true, device, true_func):

    for v in future_vals:
        t_future, state_future = extrapolate(model, t, single_true[:, 0, :], device=device, t_max=v)
        plot_sine_extrapolation(t, single_true[:, 0, :], t_future, state_future, true_func=true_func, file_name=f"sine_extrapolation ({v}).png", model=model, device=device)

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
        print("Data Loaded!")

        #load model
        model = ODEFunc(time_invariant=True).to(device)
        weights = torch.load(".\\Results\\neural_ode_sine.pth", weights_only=True)
        model.load_state_dict(weights)

        #single values
        single_true = true_traj[:, 0:1, :]

        generate_sines(model, future_vals, t, single_true, device, true_func)
