import torch
from models.neural_ode import ODEFunc, train_ode, plot_loss, extrapolate, plot_vector_field, plot_extrapolation, plot_learned_dynamics_vs_true
from dataset.sine import generate_irregular

# Device setup (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Load Data
print("Loading Data...")
t_train, y_train = generate_irregular(50)

# Sort both t and y together
sorted_indices = torch.argsort(t_train)
t_train = t_train[sorted_indices]
y_train = y_train[sorted_indices]

# Move data to device
t_train = t_train.to(device)
y_train = y_train.to(device)
print("Data Loaded!")

#Create model, optimizer and criterion
model = ODEFunc().to(device)
optimizer = torch.optim.Adam(list(model.parameters()), lr=0.005)
criterion = torch.nn.MSELoss()

#Training
print("\nTraining Neural ODE...")
losses = train_ode(model, 100, optimizer, criterion, t_train, y_train)

#Plotting loss
plot_loss(losses)
t_future, y_future = extrapolate(model, t_train, y_train, device)
plot_vector_field(model, "Vector Field for Sine Wave (100 Samples).png", device)
plot_extrapolation(t_train, y_train, t_future, y_future, "Extrapolation for Sine Wave (100 Samples).png", device)
plot_learned_dynamics_vs_true(model, device, t_train_end=4 * torch.pi, t_max=30.0, n_points=500, y_dummy_value=0.0, figsize=(10, 6), file_name="Learned Dynamics v True (100 Samples).png")
