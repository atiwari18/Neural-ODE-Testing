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


    
def train_lstm(lstm, epochs, optimizer, criterion, inputs, targets, device, file_name="lstm_sine.pth"):
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

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    full_path = os.path.join(results_dir, file_name)

    torch.save(lstm.state_dict(), full_path)

    print(f"Model saved to {full_path}")

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

#==============================================================================================================
#==============================================================================================================

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=25, num_layers=1, dropout=0.0, output_dim=2):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder = nn.LSTM(
            input_size=input_dim, 
            hidden_size = hidden_dim, 
            num_layers = num_layers, 
            batch_first = True, 
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=input_dim, 
            hidden_size = hidden_dim, 
            num_layers = num_layers, 
            batch_first = True, 
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, observed, future_len, future_truth=None, teacher_forcing_ratio=0.5):
        batch_size = observed.size(0)
        device = observed.device

        _, (hidden, cell) = self.encoder(observed)

        #start decoding from the last observed point
        decoder_input = observed[:, -1:, :]
        preds = []

        for t in range(future_len):
            decoder_out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))

            #predict the next spatial point [x, y]
            step_pred = self.output_layer(decoder_out)
            preds.append(step_pred)

            if future_truth is not None and torch.rand(1).item() < teacher_forcing_ratio:
                next_xy = future_truth[:, t:t+1, :]
            else:
                next_xy = step_pred

            last_dt = decoder_input[:, :, 2:3]

            decoder_input = torch.cat([next_xy, last_dt], dim=-1)

        preds = torch.cat(preds, dim=1)
        return preds

def split_train_test(full_data, observed_data, train_frac=0.8):
    n = full_data.size(0)
    n_train = int(train_frac * n)

    train_full = full_data[:n_train]
    test_full = full_data[n_train:]

    train_obs = observed_data[:n_train]
    test_obs = observed_data[n_train:]

    return train_full, test_full, train_obs, test_obs


def plot_rollouts(model, test_dataset, device, epoch, save_dir, plot_indices=None):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if plot_indices is None:
        plot_indices = [0, 1, 2, 3]

    valid_indices = [idx for idx in plot_indices if idx < len(test_dataset)]
    n_plot = len(valid_indices)

    observed_list = []
    future_list = []
    full_traj_list = []

    for idx in valid_indices:
        observed, future, full_traj = test_dataset[idx]
        observed_list.append(observed)
        future_list.append(future)
        full_traj_list.append(full_traj)

    observed = torch.stack(observed_list).to(device)
    future = torch.stack(future_list).to(device)
    full_traj = torch.stack(full_traj_list).to(device)

    with torch.no_grad():
        future_pred = model(observed, future_len=future.size(1), future_truth=None, teacher_forcing_ratio=0.0)

    observed = observed.cpu().numpy()
    future = future.cpu().numpy()
    full_traj = full_traj.cpu().numpy()
    future_pred = future_pred.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for panel_i, sample_i in enumerate(valid_indices):
        ax = axes[panel_i]

        #Observed input includes delta_t but only x, y need to be plotted
        obs = observed[panel_i][:, :2]
        true_full = full_traj[panel_i]
        pred_full = np.concatenate([obs, future_pred[panel_i]], axis=0)

        ax.plot(true_full[:, 0], true_full[:, 1], "k--", linewidth=1.5, label="true full traj")
        ax.plot(pred_full[:, 0], pred_full[:, 1], color="red", linewidth=2, label="predicted rollout")
        ax.scatter(obs[:, 0], obs[:, 1], color="blue", s=10, label="observed prefix")

        ax.scatter(obs[0, 0], obs[0, 1], color="green", s=40, label="start")
        ax.scatter(obs[-1, 0], obs[-1, 1], color="orange", s=40, label="obs end")
        ax.scatter(true_full[-1, 0], true_full[-1, 1], color="purple", s=40, label="target end")

        ax.set_title(f"Trajectory {sample_i}")
        ax.axis("equal")

    for j in range(n_plot, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"LSTM Spiral Extrapolation Epoch {epoch:04d}", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])

    save_path = os.path.join(save_dir, f"lstm_spiral_epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {save_path}")
