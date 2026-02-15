import torch
from dataset.data import SineDynamics, generate_sine
from models.lstm import LSTM, train_lstm, plot_lstm_sine_extrapolation

# Device setup (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Load Data
print("Loading Data...")
true_func = SineDynamics(device=device).to(device)
t, y0, true_traj = generate_sine(true_func, batch_size=16, n_samples=100, t_max=4*torch.pi, device=device)
print(f"Trajectories shape: {true_traj.shape}")

#Create LSTM
seq_len = 20
model = LSTM(input_dim=2, hidden_dim=64, num_layers=2, output_dim=2).to(device)

print("\nTraining LSTM...")

