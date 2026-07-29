import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
import os
import lib.utils as utils
from lib.base_models import Baseline, VAE_Baseline

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

def split_train_val_test(full_data, observed_data, train_frac=0.7, val_frac=0.15):
    n = full_data.size(0)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_full = full_data[:n_train]
    val_full = full_data[n_train:n_train + n_val]
    test_full = full_data[n_train + n_val:]

    train_obs = observed_data[:n_train]
    val_obs = observed_data[n_train:n_train + n_val]
    test_obs = observed_data[n_train + n_val:]

    return train_full, val_full, test_full, train_obs, val_obs, test_obs

def get_plot_grid(n_plot):
    n_cols = int(np.ceil(np.sqrt(n_plot)))
    n_rows = int(np.ceil(n_plot / n_cols))
    return n_rows, n_cols

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
        future_pred = model(
            observed,
            future_len=future.size(1),
            future_truth=None,
            teacher_forcing_ratio=0.0,
        )

    future_mse = ((future_pred - future) ** 2).mean(dim=(1, 2)).cpu().numpy()

    observed = observed.cpu().numpy()
    full_traj = full_traj.cpu().numpy()
    future_pred = future_pred.cpu().numpy()

    n_rows, n_cols = get_plot_grid(n_plot)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for panel_i, sample_i in enumerate(valid_indices):
        ax = axes[panel_i]

        # Observed tensor includes [x, y, delta_t], but we only plot x/y.
        obs_xy = observed[panel_i][:, :2]
        true_full = full_traj[panel_i]

        # Start predicted future from the last observed point.
        future_rollout = np.concatenate([obs_xy[-1:, :], future_pred[panel_i]], axis=0)

        ax.plot(
            true_full[:, 0],
            true_full[:, 1],
            "k--",
            linewidth=1.5,
            label="true full traj",
        )

        ax.plot(
            future_rollout[:, 0],
            future_rollout[:, 1],
            color="red",
            linewidth=2,
            label="predicted future",
        )

        ax.scatter(
            obs_xy[:, 0],
            obs_xy[:, 1],
            color="blue",
            s=12,
            label="observed samples",
        )

        # Green marker is the first observed irregular point.
        ax.scatter(
            obs_xy[0, 0],
            obs_xy[0, 1],
            color="green",
            s=40,
            label="first obs",
        )

        # Orange marker is the last observed irregular point.
        ax.scatter(
            obs_xy[-1, 0],
            obs_xy[-1, 1],
            color="orange",
            s=40,
            label="last obs",
        )

        # Purple marker is the end of the dense target trajectory.
        ax.scatter(
            true_full[-1, 0],
            true_full[-1, 1],
            color="purple",
            s=40,
            label="target end",
        )

        ax.set_title(f"Trajectory {sample_i} | Future MSE = {future_mse[panel_i]:.4f}")
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

####################################################################################################
####################################################################################################
#Mase aware autoregressive LSTM baseline
#INPUT: [values, feature_mask, elapsed delta_t]
class LSTMDeltaT(Baseline):
    def __init__(self, input_dim, hidden_dim, device, obsrv_std=0.01, n_units=50):
        super().__init__(
            input_dim=input_dim,
            latent_dim=hidden_dim,
            device=device,
            obsrv_std=obsrv_std,
        )

        self.hidden_dim = hidden_dim

        self.lstm_cell = nn.LSTMCell(input_size=input_dim * 2 + 1, hidden_size=hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, input_dim),
        )

        utils.init_network_weights(self.decoder)

    def get_reconstruction(
        self,
        time_steps_to_predict,
        data,
        truth_time_steps,
        mask=None,
        n_traj_samples=1,
        mode=None,
    ):
        if mask is None:
            raise ValueError("LSTMDeltaT requires an observation mask")

        if mode == "extrap":
            raise ValueError(
                "Autoregressive LSTMDeltaT is used for "
                "interpolation only"
            )

        if (
            len(time_steps_to_predict) != len(truth_time_steps)
            or not torch.allclose(
                time_steps_to_predict,
                truth_time_steps,
            )
        ):
            raise ValueError(
                "LSTMDeltaT expects identical observed and "
                "prediction timelines"
            )

        batch_size, n_timepoints, _ = data.shape
        device = data.device

        hidden = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device,
        )

        cell = torch.zeros_like(hidden)

        # Time since each patient's previous actual observation.
        elapsed = torch.zeros(
            batch_size,
            1,
            device=device,
        )

        hidden_states = []

        for index in range(n_timepoints):
            if index > 0:
                shared_delta = (
                    truth_time_steps[index]
                    - truth_time_steps[index - 1]
                )

                elapsed = elapsed + shared_delta

            values_i = data[:, index, :]
            mask_i = mask[:, index, :]

            has_observation = (
                mask_i.sum(dim=-1, keepdim=True) > 0
            ).to(data.dtype)

            lstm_input = torch.cat(
                [values_i, mask_i, elapsed],
                dim=-1,
            )

            proposed_hidden, proposed_cell = self.lstm_cell(
                lstm_input,
                (hidden, cell),
            )

            # Do not update for timestamps belonging only to another
            # patient in the batch.
            hidden = (
                has_observation * proposed_hidden
                + (1.0 - has_observation) * hidden
            )

            cell = (
                has_observation * proposed_cell
                + (1.0 - has_observation) * cell
            )

            elapsed = torch.where(
                has_observation.bool(),
                torch.zeros_like(elapsed),
                elapsed,
            )

            hidden_states.append(hidden)

        hidden_states = torch.stack(hidden_states, dim=1)

        # [1, batch, time, input_dim]
        outputs = self.decoder(hidden_states).unsqueeze(0)

        # Match the existing autoregressive ODE-RNN convention:
        # output at t_i is based on state from t_(i-1).
        outputs = utils.shift_outputs(outputs, first_datapoint=data[:, 0, :])

        zero_std = torch.zeros(1, batch_size, self.hidden_dim, device=device)

        final_hidden = hidden.unsqueeze(0)

        extra_info = {
            "first_point": (
                final_hidden,
                zero_std,
                final_hidden,
            )
        }

        return outputs, extra_info

class LSTMDeltaTVAE(VAE_Baseline):
    """
    Variational LSTM encoder-decoder baseline.

    Encoder:
        masked values + masks + accumulated delta_t
        -> approximate posterior q(z0 | observations)

    Decoder:
        sampled z0
        -> autoregressive LSTM reconstruction/prediction
    """
    def __init__(self, input_dim, latent_dim, rec_dims, z0_prior, device, obsrv_std=0.01, n_units=50):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            z0_prior=z0_prior,
            device=device,
            obsrv_std=obsrv_std,
        )

        self.rec_dims = rec_dims

        self.encoder_cell = nn.LSTMCell(
            input_size=input_dim * 2 + 1,
            hidden_size=rec_dims,
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(rec_dims, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2),
        )

        self.z_to_hidden = nn.Linear(
            latent_dim,
            latent_dim,
        )

        self.z_to_cell = nn.Linear(
            latent_dim,
            latent_dim,
        )

        self.decoder_cell = nn.LSTMCell(
            input_size=input_dim * 2 + 1,
            hidden_size=latent_dim,
        )

        self.output_net = nn.Sequential(
            nn.Linear(latent_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, input_dim),
        )

        utils.init_network_weights(self.posterior_net)
        utils.init_network_weights(self.output_net)

    def _encode(
        self,
        data,
        mask,
        time_steps,
        run_backwards,
    ):
        batch_size, n_timepoints, _ = data.shape
        device = data.device

        hidden = torch.zeros(
            batch_size,
            self.rec_dims,
            device=device,
        )

        cell = torch.zeros_like(hidden)

        elapsed = torch.zeros(
            batch_size,
            1,
            device=device,
        )

        if run_backwards:
            indices = list(range(n_timepoints - 1, -1, -1))
        else:
            indices = list(range(n_timepoints))

        previous_time = None

        for index in indices:
            current_time = time_steps[index]

            if previous_time is not None:
                elapsed = elapsed + torch.abs(
                    current_time - previous_time
                )

            values_i = data[:, index, :]
            mask_i = mask[:, index, :]

            has_observation = (
                mask_i.sum(dim=-1, keepdim=True) > 0
            ).to(data.dtype)

            encoder_input = torch.cat(
                [values_i, mask_i, elapsed],
                dim=-1,
            )

            proposed_hidden, proposed_cell = self.encoder_cell(
                encoder_input,
                (hidden, cell),
            )

            hidden = (
                has_observation * proposed_hidden
                + (1.0 - has_observation) * hidden
            )

            cell = (
                has_observation * proposed_cell
                + (1.0 - has_observation) * cell
            )

            elapsed = torch.where(
                has_observation.bool(),
                torch.zeros_like(elapsed),
                elapsed,
            )

            previous_time = current_time

        posterior_params = self.posterior_net(hidden)
        z0_mean, z0_raw_std = torch.chunk(
            posterior_params,
            chunks=2,
            dim=-1,
        )

        z0_std = torch.nn.functional.softplus(z0_raw_std) + 1e-8

        return z0_mean, z0_std

    def _last_observed_values(self, data, mask):
        batch_size, n_timepoints, _ = data.shape
        device = data.device

        observed_event = mask.sum(dim=-1) > 0

        indices = torch.arange(
            n_timepoints,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1)

        last_indices = torch.where(
            observed_event,
            indices,
            torch.zeros_like(indices),
        ).max(dim=1).values

        batch_indices = torch.arange(batch_size, device=device)

        return data[batch_indices, last_indices]

    def _decode(
        self,
        z_samples,
        data,
        mask,
        truth_time_steps,
        time_steps_to_predict,
        mode,
    ):
        n_samples, batch_size, latent_dim = z_samples.shape
        target_length = len(time_steps_to_predict)
        input_dim = data.size(-1)

        flattened_z = z_samples.reshape(
            n_samples * batch_size,
            latent_dim,
        )

        hidden = self.z_to_hidden(flattened_z)
        cell = self.z_to_cell(flattened_z)

        if mode == "extrap":
            seed_values = self._last_observed_values(data, mask)

            previous_values = seed_values.repeat(n_samples, 1)

            first_delta = (time_steps_to_predict[0] - truth_time_steps[-1])

            decoder_deltas = torch.cat(
                [first_delta.reshape(1), time_steps_to_predict[1:] - time_steps_to_predict[:-1]], dim=0)

            predictions = []
            start_index = 0

        else:
            first_values = data[:, 0, :]

            previous_values = first_values.repeat(n_samples, 1)

            # As in the existing RNN-VAE implementation, the first
            # interpolation output is the first observed point.
            predictions = [
                previous_values.reshape(
                    n_samples,
                    batch_size,
                    input_dim,
                )
            ]

            decoder_deltas = torch.cat(
                [
                    torch.zeros(
                        1,
                        device=data.device,
                        dtype=data.dtype,
                    ),
                    time_steps_to_predict[1:]
                    - time_steps_to_predict[:-1],
                ],
                dim=0,
            )

            start_index = 1

        generated_mask = torch.ones_like(previous_values)

        for index in range(start_index, target_length):
            delta_t = decoder_deltas[index].expand(
                n_samples * batch_size,
                1,
            )

            decoder_input = torch.cat(
                [
                    previous_values,
                    generated_mask,
                    delta_t,
                ],
                dim=-1,
            )

            hidden, cell = self.decoder_cell(
                decoder_input,
                (hidden, cell),
            )

            predicted_values = self.output_net(hidden)

            predictions.append(
                predicted_values.reshape(
                    n_samples,
                    batch_size,
                    input_dim,
                )
            )

            previous_values = predicted_values

        return torch.stack(predictions, dim=2)

    def get_reconstruction(
        self,
        time_steps_to_predict,
        data,
        truth_time_steps,
        mask=None,
        n_traj_samples=1,
        mode=None,
    ):
        if mask is None:
            raise ValueError(
                "LSTMDeltaTVAE requires an observation mask"
            )

        if mode not in {"interp", "extrap"}:
            raise ValueError(
                f"Unknown reconstruction mode: {mode}"
            )

        # Match the encoder-decoder protocol:
        # interpolation -> encode backward toward t0
        # extrapolation -> encode forward toward the last observation
        run_backwards = mode == "interp"

        z0_mean, z0_std = self._encode(
            data,
            mask,
            truth_time_steps,
            run_backwards=run_backwards,
        )

        n_traj_samples = max(1, int(n_traj_samples))

        repeated_mean = z0_mean.unsqueeze(0).repeat(n_traj_samples, 1, 1)

        repeated_std = z0_std.unsqueeze(0).repeat(n_traj_samples, 1, 1)

        z_samples = utils.sample_standard_gaussian(repeated_mean, repeated_std)

        outputs = self._decode(
            z_samples,
            data,
            mask,
            truth_time_steps,
            time_steps_to_predict,
            mode,
        )

        extra_info = {
            "first_point": (
                z0_mean.unsqueeze(0),
                z0_std.unsqueeze(0),
                z_samples,
            )
        }

        return outputs, extra_info