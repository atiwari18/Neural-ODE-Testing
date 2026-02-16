import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
import os

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=2):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        #lstm layers
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )

        #output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        return output, hidden
    
    def rollout(self, seed_sequence, t_train, t_max, device):
        #computr timestep frm training data and derive n_steps from t_max
        dt = (t_train[1] - t_train[0]).item()
        t_start = t_train[-1].item()
        n_future_steps = int((t_max - t_start) / dt)               #How many steps to predict up yto t_max
        n_train_steps = len(t_train) - len(seed_sequence[0])

        #future time points
        t_future = torch.linspace(t_start, t_max, n_future_steps).to(device)
        t_all = torch.cat([t_train, t_future[1:]])

        self.eval()
        predictions =[]

        with torch.no_grad():
            #process seed sequence to build hiddn state, lstm stores memory in the hidden state
            #that memory is needed to continue with predictions
            seed_out, hidden = self.forward(seed_sequence)

            # Add seed predictions to output
            for j in range(seed_out.shape[1]):
                predictions.append(seed_out[:, j, :])              #Last prediction from seed

            #start from laast obsercvation
            current_input = seed_out[:, -1:, :]

            for _ in range(n_train_steps):
                output, hidden = self.forward(current_input, hidden)
                predictions.append(output[:, 0, :])
                current_input = output

            #Continue into future region
            for _ in range(n_future_steps - 1):
                output, hidden = self.forward(current_input, hidden)
                predictions.append(output[:, 0, :])
                current_input = output

        #stack the preds
        return torch.stack(predictions, dim=0), t_all


    
def train_lstm(lstm, epochs, optimizer, criterion, inputs, targets, device):
    n_windows = inputs.shape[0]
    losses = []

    #training loop
    lstm.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        #shuffle order of windows each epoch
        perm = torch.randperm(n_windows)

        for i in perm:
            #get batch
            x = inputs[i].to(device)
            y = targets[i].to(device)

            optimizer.zero_grad()

            #forward pass
            pred, _ = lstm(x)

            #loss
            loss = criterion(pred, y)

            #backwards
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        #calculate the average loss
        avg_loss = epoch_loss / n_windows
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    return losses
    

def plot_lstm_sine_extrapolation(t_train, state_train, t_all, lstm_all, true_func=None, t_max=None, file_name="lstm_sine_extrapolation.png", device="cpu"):
        #Full train
        t_train_np = t_train.cpu().numpy()
        y_train = state_train[:, 0].cpu().numpy()
        y0_train = state_train[0:1, :].to(device)

        #Full LSTM trajectory
        t_all_np = t_all.cpu().numpy()
        y_all = lstm_all[:, 0].cpu().numpy()  # Position only

        plt.figure(figsize=(14, 6))

        # Ground truth extended to t_max so it covers both training AND extrapolation
        if true_func is not None:
            # Use t_max if provided, otherwise just cover training region
            gt_end = t_max if t_max is not None else t_train[-1].item()

            with torch.no_grad():
                t_gt = torch.linspace(t_train[0].item(), gt_end, 500).to(device)
                state_gt = odeint(true_func, y0_train, t_gt)
                plt.plot(t_gt.cpu().numpy(), state_gt[:, 0, 0].cpu().numpy(),
                        'gray', linestyle='--', alpha=0.5, linewidth=2.5,
                        label='True Dynamics')

        # Single continuous LSTM line from training start to t_max
        plt.plot(t_all_np[:len(y_all)], y_all, 'green', linewidth=2.5, alpha=0.8,
                label='LSTM Trajectory')

        # Training observations
        plt.scatter(t_train_np, y_train, c='red', s=40, alpha=0.7,
                zorder=5, label='Training Observations')

        # Mark boundary between training and extrapolation
        plt.axvline(x=t_train_np[-1], color='orange', linestyle=':',
                linewidth=2, alpha=0.7, label='End of Training')

        plt.title("LSTM: Sine Wave Extrapolation", fontsize=14, fontweight='bold')
        plt.xlabel("Time (t)", fontsize=12)
        plt.ylabel("Position (y)", fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)

        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        project_root = os.path.dirname(script_dir)
        results_dir = os.path.join(project_root, 'Results')
        os.makedirs(results_dir, exist_ok=True)

        plt.savefig(os.path.join(results_dir, file_name), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {os.path.join(results_dir, file_name)}")