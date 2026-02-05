import torch
from models.neural_ode import ODEFunc, train_ode, plot_loss, extrapolate, plot_vector_field, plot_extrapolation, plot_learned_dynamics_vs_true
from dataset.data import generate_irregular

# Device setup (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Data
print("Loading Data...")
t_train, state_train = generate_irregular(100)

# Sort by time and keep state aligned
sorted_indices = torch.argsort(t_train)
t_train = t_train[sorted_indices]
state_train = state_train[sorted_indices]

# Move data to device
t_train = t_train.to(device)
state_train = state_train.to(device)
print("Data Loaded!")
print(f"State shape: {state_train.shape}")  # Should be [100, 2]

# Create model, optimizer and criterion
model = ODEFunc(hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Training
print("\nTraining Neural ODE...")
losses = train_ode(model, 100, optimizer, criterion, t_train, state_train)

# Plotting
print("\nGenerating plots...")
plot_loss(losses)

t_future, state_future = extrapolate(model, t_train, state_train, device)

plot_vector_field(model, file_name="Learned Vector Field (time agnostic-2nd-run).png", device=device)

plot_extrapolation(t_train, state_train, t_future, state_future, "Extrapolation for Sine Wave (time agnostic-2nd-run).png", model, device)

plot_learned_dynamics_vs_true(model, device=device, file_name="Learned Dynamics (time agnostic-2nd-run).png")
