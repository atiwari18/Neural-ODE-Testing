from torch.utils.data import Dataset
import torch

#Dataset for irregular spirals
class SpiralLTCDataset(Dataset):
    def __init__(self, observed_data, full_data, full_tp, observed_tp, observed_offsets):
        self.observed_data = observed_data          # [N, obs_len, 2]
        self.full_data = full_data                  # [N, pred_len, 2]
        self.full_tp = full_tp                      # [pred_len]
        self.observed_tp = observed_tp              # [obs_len]
        self.observed_offsets = observed_offsets    # dense indices of observed points

        # Last observed point in the dense full trajectory
        self.last_obs_dense_idx = int(observed_offsets[-1].item())

        # Compute delta-t for the observed irregular prefix
        # First step uses time since local t=0
        observed_delta_t = observed_tp.clone()
        observed_delta_t[1:] = observed_tp[1:] - observed_tp[:-1]
        self.observed_delta_t = observed_delta_t

        # Future starts immediately after the last observed dense index
        self.future_tp = full_tp[self.last_obs_dense_idx + 1:]

        # Compute delta-t for each future prediction step
        previous_tp = torch.cat(
            [full_tp[self.last_obs_dense_idx:self.last_obs_dense_idx + 1], self.future_tp[:-1]],
            dim=0,
        )
        self.future_delta_t = self.future_tp - previous_tp
        self.future_delta_t = self.future_delta_t

    def __len__(self):
        return self.observed_data.size(0)
    
    def __getitem__(self, index):
        observed_xy = self.observed_data[index]
        full_traj = self.full_data[index]

        #Dense future target starts after the last observed dense point
        future_xy = full_traj[self.last_obs_dense_idx + 1:]

        return (observed_xy, self.observed_delta_t, future_xy, self.future_delta_t, full_traj)
    
def split_train_test(full_data, observed_data, train_frac=0.8):
    #Get total number of trajectories
    n = full_data.size(0)

    #Compute how many samples go to training split
    n_train = int(train_frac * n)

    #Take the first chunk for training targets (full dense trajectories).
    train_full = full_data[:n_train]

    #tke the remaining chunk for test targets.
    test_full = full_data[n_train:]

    #take the first chunk for training observations (irregular observed prefixes).
    train_obs = observed_data[:n_train]

    # ake the remaining chunk for test observations.
    test_obs = observed_data[n_train:]

    return train_full, test_full, train_obs, test_obs
