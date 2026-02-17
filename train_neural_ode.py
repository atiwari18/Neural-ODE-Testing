import torch
from models.neural_ode import ODEFunc, train_ode, plot_loss, extrapolate, plot_comparison, plot_sine_extrapolation
from dataset.data import SineDynamics, generate_sine

# Device setup (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Data
print("Loading Data...")
true_func = SineDynamics(device=device).to(device)
t, y0, true_traj = generate_sine(true_func, batch_size=16, n_samples=100, t_max=4*torch.pi, device=device)
t = t.to(device)
y0 = y0.to(device)
true_traj = true_traj.to(device)
print("Data Loaded!")
print(f"State shape: y{y0.shape}") 

# Create model, optimizer and criterion
model = ODEFunc(time_invariant=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training
print("\nTraining Neural ODE...")
losses =  train_ode(model, 1000, optimizer, criterion, true_traj=true_traj, t=t, y0=y0, file_name="neural_ode_sine_1000.pth")

# Plotting
print("\nGenerating plots...")
plot_loss(losses, file_name="Losses (sine-1000).png")

#3xtrapolating one
single_y0 = y0[0:1]
single_true = true_traj[:, 0:1, :]
t_future, state_future = extrapolate(model, t, single_true[:, 0, :], device=device, t_max=6*torch.pi)
plot_sine_extrapolation(t, single_true[:, 0, :], t_future, state_future, true_func=true_func, file_name="single_extrapolation (sine-1000).png", model=model, device=device)
