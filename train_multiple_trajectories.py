import torch
from models.neural_ode import ODEFunc, train_ode, plot_loss, extrapolate, plot_spiral_extrapolation, plot_comparison
from dataset.data import SpiralDynamics, generate_spiral

# Device setup (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading Data...")
true_func = SpiralDynamics(device=device).to(device)
t, y0, true_traj = generate_spiral(true_func, batch_size=16, n_samples=100, t_max=10, device=device)
t = t.to(device)
y0 = y0.to(device)
true_traj = true_traj.to(device)
print("Data Loaded!")
print(f"State shape: y{y0.shape}") 

#MODEL, CRITERION AND OPTIMIZER
model = ODEFunc(time_invariant=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

#Training
print("\nTraining Neural ODE...")
losses =  train_ode(model, 100, optimizer, criterion, true_traj=true_traj, t=t, y0=y0)

print("\nGenerating Visualizations...")
plot_loss(losses, file_name="Losses (multiple-traj-100).png")
plot_comparison(true_func, model, device=device, file_name="spiral_comparision (multiple-traj-100).png")

#3xtrapolating one
single_y0 = y0[0:1]
single_true = true_traj[:, 0:1, :]
t_future, state_future = extrapolate(model, t, single_true[:, 0, :], device=device, t_max=30)
plot_spiral_extrapolation(t, single_true[:, 0, :], state_future, true_func=true_func, file_name="single_extrapolation (multiple-traj-100).png", model=model, device=device)




