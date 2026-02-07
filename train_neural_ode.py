import torch
from models.neural_ode import ODEFunc, train_ode, plot_loss, extrapolate, plot_vector_field, plot_extrapolation, plot_learned_dynamics_vs_true, plot_spiral_extrapolation
from dataset.data import generate_irregular, generate_spiral

# Device setup (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Data
print("Loading Data...")
#t_train, state_train = generate_irregular(100)
data = generate_spiral(100, t_max=10)
y0 = torch.tensor([1.0, 0.0])
t = torch.linspace(0, 10, 100)

# Sort by time and keep state aligned
# sorted_indices = torch.argsort(t_train)
# t_train = t_train[sorted_indices]
# state_train = state_train[sorted_indices]

# # Move data to device
# t_train = t_train.to(device)
# state_train = state_train.to(device)
data = data.to(device)
y0 = y0.to(device)
t = t.to(device)

print("Data Loaded!")
print(f"State shape: y{y0.shape}")  # Should be [100, 2]

# Create model, optimizer and criterion
model = ODEFunc(hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Training
print("\nTraining Neural ODE...")
losses = train_ode(model, 100, optimizer, criterion, t, data)

# Plotting
print("\nGenerating plots...")
plot_loss(losses, file_name="Losses (spiral-time-conditioned).png")

t_future, state_future = extrapolate(model, t, data, device, t_max=20)

#plot_vector_field(model, file_name="Learned Vector Field (spiral-dataset).png", device=device)

#plot_extrapolation(t, data, t_future, state_future, "Extrapolation for Sine Wave (spiral-dataset).png", model, device)

#plot_learned_dynamics_vs_true(model, device=device, file_name="Learned Dynamics (spiral-dataset).png")

plot_spiral_extrapolation(t, data, state_future, file_name="Spiral Extrapolation (spiral-time-conditioned).png")
