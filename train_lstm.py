import torch
from models.neural_ode import plot_loss
from dataset.data import SineDynamics, generate_sine
from dataset.lstm_dataset import generate_lstm_dataset
from models.lstm import LSTM, train_lstm, plot_lstm_sine_extrapolation

# Device setup (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Load Data
print("Loading Data...")
true_func = SineDynamics(device=device).to(device)
t, y0, true_traj = generate_sine(true_func, batch_size=16, n_samples=100, t_max=4*torch.pi, device=device)
inputs, targets = generate_lstm_dataset(true_traj, seq_len=20)

print(f"Trajectories shape: {true_traj.shape}")

#Create LSTM
seq_len = 20
model = LSTM(input_dim=2, hidden_dim=64, num_layers=2, output_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

print("\nTraining LSTM...")
losses = train_lstm(model, 50, optimizer, criterion, inputs, targets, device)

print("\nGenerating plots...")
plot_loss(losses, file_name="LSTM Losses (sine-50).png")

#Format of tensor --> tensor[time_index, batch_index, feature_index]
single_traj = true_traj[:, 0, :]                                    #[all timesteps, first trajectory, take all features]
seed = single_traj[:seq_len].unsqueeze(0).to(device)                #single_traj[:the first 20 timesteps].unsqueeze(0) = [seq_len, input_dim] --> [1, seq_len, input_dim]

lstm_all, t_all = model.rollout(seed, t_train=t, t_max=6 * torch.pi, device=device)
lstm_all = lstm_all[:, 0, :]

#Plot with ground truth extended to t_max
plot_lstm_sine_extrapolation(t_train=t, state_train=single_traj, 
                             t_all=t_all, lstm_all=lstm_all, 
                             true_func=true_func, t_max=6 * torch.pi, file_name="lstm_sine_extrapolation (sine-50).png", device=device)