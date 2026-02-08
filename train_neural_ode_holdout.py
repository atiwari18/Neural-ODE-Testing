import torch
from models.neural_ode import ODEFunc, train_ode, plot_loss, extrapolate, plot_spiral_extrapolation, evaluate_on_holdout, plot_train_test_comparison
from dataset.data import generate_spiral

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

# Load Data
print("Loading Data...")
# Generate full dataset
n_samples = 200
t_max = 6.29*5
data_full = generate_spiral(n_samples, t_max=t_max)
t_full = torch.linspace(0, t_max, n_samples)

# Create train/test split (80/20 split, randomly sampled from same time range)
train_ratio = 0.8
n_train = int(n_samples * train_ratio)

# Random shuffle indices
torch.manual_seed(42)  # For reproducibility
indices = torch.randperm(n_samples)
train_indices = indices[:n_train]
test_indices = indices[n_train:]

# Sort train indices to maintain temporal order for ODE solver
train_indices_sorted = torch.sort(train_indices)[0]
test_indices_sorted = torch.sort(test_indices)[0]

# Split data
t_train = t_full[train_indices_sorted]
data_train = data_full[train_indices_sorted]

t_test = t_full[test_indices_sorted]
data_test = data_full[test_indices_sorted]

# Move data to device
t_train = t_train.to(device)
data_train = data_train.to(device)
t_test = t_test.to(device)
data_test = data_test.to(device)

print("Data Loaded!")
print(f"Train samples: {len(t_train)} | Test samples: {len(t_test)}")
print(f"Train time range: [{t_train[0]:.2f}, {t_train[-1]:.2f}]")
print(f"Test time range: [{t_test[0]:.2f}, {t_test[-1]:.2f}]")
print(f"Data shape: {data_train.shape}")

# Create model, optimizer and criterion
model = ODEFunc(hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.MSELoss()

# Training
print("\nTraining Neural ODE...")
losses = train_ode(model, 100, optimizer, criterion, t_train, data_train)

# Evaluate on hold-out test set
print("\nEvaluating on hold-out test set...")
test_loss = evaluate_on_holdout(model, criterion, t_test, data_test)
print(f"Test Loss (MSE): {test_loss:.6f}")
print(f"Final Train Loss: {losses[-1]:.6f}")

# Plotting
print("\nGenerating plots...")
plot_loss(losses, file_name="Losses_with_holdout (small-run-100-epochs).png")

# Plot train vs test comparison
plot_train_test_comparison(
    model, 
    t_train, data_train, 
    t_test, data_test, 
    t_max, 
    device,
    file_name="Train_Test_Comparison (small-run-100-epochs).png"
)

# Extrapolation beyond training range
t_future, state_future = extrapolate(model, t_train, data_train, device, t_max=30)

plot_spiral_extrapolation(
    t_train, 
    data_train, 
    state_future, 
    file_name="Spiral_Extrapolation_with_holdout (small-run-100-epochs).png", 
    model=model, 
    device = device
)
