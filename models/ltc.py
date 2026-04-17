import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from ncps.torch import LTC
from ncps import wirings

class Seq2SeqLTC(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2, mixed_memory=False, ode_unfolds=6, 
                 use_ncp=False, sparsity_level=0.5, seed=12):
        super().__init__()

        #If use_ncp is enables then create a structured sparse NCP wiring
        if use_ncp:
            wiring = wirings.AutoNCP(
                units=hidden_dim, 
                output_size=output_dim, 
                sparsity_level=sparsity_level, 
                seed=seed
            )

            ltc_units = wiring
        
        else:
            ltc_units = hidden_dim

        #LTC reccurent backbone
        self.rnn = LTC(
            input_size=input_dim, 
            units=ltc_units, 
            return_sequences=False, 
            batch_first=True, 
            mixed_memory=mixed_memory, 
            ode_unfolds=ode_unfolds
        )

        #For NCP, the motor layer already has output_dim, so a linear layer is optional
        if use_ncp:
            self.output_layer = nn.Identity()
        else:
            #Map LTC outputs back to x, y coordinates
            self.output_layer = nn.Linear(self.rnn.output_size, output_dim)

        self.use_ncp = use_ncp

    def forward(self, observed_xy, observed_dt, future_dt, future_truth=None, teacher_forcing_ratio=0.0):
        shared_observed_dt = observed_dt[0:1, :]
        
        #Encode the observed irregular sequence, timespans tells LTC
        #how much time has elapsed before each step.
        _, hidden = self.rnn(observed_xy, timespans=shared_observed_dt)

        #Start decoding from the last observed spatial point
        decoder_input = observed_xy[:, -1:, :]
        preds = []

        shared_future_dt = future_dt[0:1, :]

        #Predict one future step at a time
        for step in range(shared_future_dt.size(1)):
            step_dt = shared_future_dt[:, step:step+1]

            #Advance the ltc state using the elapsed time
            decoder_out, hidden = self.rnn(
                decoder_input, 
                hx=hidden, 
                timespans=step_dt
            )

            #Convert LTC output to a 2d
            step_pred = self.output_layer(decoder_out).unsqueeze(1)
            preds.append(step_pred)

            #Teacher forcing: sometimes feed the true next point during training
            if future_truth is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = future_truth[:, step:step+1, :]
            else:
                decoder_input = step_pred

        return torch.cat(preds, dim=1)
    
def plot_ltc_rollouts(model, test_dataset, device, epoch, save_dir, plot_indices=None):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if plot_indices is None:
        plot_indices = [0, 1, 2, 3]

    valid_indices = [idx for idx in plot_indices if idx < len(test_dataset)]

    if not valid_indices:
        return

    observed_xy_list = []
    observed_dt_list = []
    future_xy_list = []
    future_dt_list = []
    full_traj_list = []

    for idx in valid_indices:
        #Each LTC sample contains:
        #observed_xy  -> irregular observed prefix
        #observed_dt  -> elapsed times between observed points
        #future_xy    -> dense future target we want to predict
        #future_dt    -> elapsed times for future rollout steps
        #full_traj    -> full dense trajectory for plotting/reference
        observed_xy, observed_dt, future_xy, future_dt, full_traj = test_dataset[idx]

        # Store each part in its corresponding list.
        observed_xy_list.append(observed_xy)
        observed_dt_list.append(observed_dt)
        future_xy_list.append(future_xy)
        future_dt_list.append(future_dt)
        full_traj_list.append(full_traj)

    # Stack the selected individual samples into batched tensors.
    observed_xy = torch.stack(observed_xy_list).to(device)
    observed_dt = torch.stack(observed_dt_list).to(device)
    future_xy = torch.stack(future_xy_list).to(device)
    future_dt = torch.stack(future_dt_list).to(device)
    full_traj = torch.stack(full_traj_list).to(device)

    # Disable gradient tracking because we are only evaluating/plotting.
    with torch.no_grad():
        # Ask the LTC to predict the future trajectory from the observed prefix.
        future_pred = model(
            observed_xy,
            observed_dt,
            future_dt,
            future_truth=None,
            teacher_forcing_ratio=0.0,
        )

    # Compute one future-horizon MSE value per plotted sample.
    future_mse = ((future_pred - future_xy) ** 2).mean(dim=(1, 2)).cpu().numpy()

    # Move tensors to CPU and convert to NumPy so matplotlib can plot them.
    observed_xy = observed_xy.cpu().numpy()
    full_traj = full_traj.cpu().numpy()
    future_pred = future_pred.cpu().numpy()

    # Create a 2x2 grid of subplots, matching the style of your other spiral plots.
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Flatten the axes array so we can index it with a single integer.
    axes = axes.flatten()

    # Plot each requested sample in its own panel.
    for panel_i, sample_i in enumerate(valid_indices):
        # Select the current axis.
        ax = axes[panel_i]

        # Get the observed prefix for this plotted sample.
        obs = observed_xy[panel_i]

        # Get the full ground-truth dense trajectory for this sample.
        true_full = full_traj[panel_i]

        # Join the last observed point to the predicted future so the rollout
        # appears visually continuous from observation to extrapolation.
        pred_future = np.concatenate([obs[-1:, :], future_pred[panel_i]], axis=0)

        # Plot the full ground-truth trajectory as a dashed black reference curve.
        ax.plot(true_full[:, 0], true_full[:, 1], "k--", linewidth=1.5, label="true full traj")

        # Plot the LTC-predicted future rollout in red.
        ax.plot(pred_future[:, 0], pred_future[:, 1], color="red", linewidth=2, label="predicted future")
        
        # Plot the observed irregular samples as blue points.
        ax.scatter(obs[:, 0], obs[:, 1], color="blue", s=12, label="observed samples")

        # Mark the first observed point in green.
        ax.scatter(obs[0, 0], obs[0, 1], color="green", s=40, label="first obs")

        # Mark the last observed point in orange.
        ax.scatter(obs[-1, 0], obs[-1, 1], color="orange", s=40, label="last obs")

        # Mark the final target point of the full dense trajectory in purple.
        ax.scatter(true_full[-1, 0], true_full[-1, 1], color="purple", s=40, label="target end")

        # Add the sample index and its future prediction error to the title.
        ax.set_title(f"Trajectory {sample_i} | Future MSE = {future_mse[panel_i]:.4f}")

        # Keep x and y scales equal so spiral geometry is not distorted.
        ax.axis("equal")

    # If fewer than four panels are used, hide the unused axes.
    for panel_i in range(len(valid_indices), len(axes)):
        axes[panel_i].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"LTC Spiral Extrapolation Epoch {epoch:04d}", y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    
    save_path = os.path.join(save_dir, f"ltc_spiral_epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Print where the plot was saved so training logs show the output path.
    print(f"Saved plot to {save_path}")