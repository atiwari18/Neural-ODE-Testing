import torch
from torch.utils.data import Dataset

#Converts trajectories into input/target sequences
def generate_lstm_dataset(trajectories, seq_len=20):
    #rearrange to be [batch_size, n_points, 2]
    traj = trajectories.permute(1, 0, 2)
    
    n_points = traj.shape[1]
    inputs, targets = [], []

    #slide a window over the trajectory
    for i in range(n_points - seq_len):
        inputs.append(traj[:, i:i+seq_len, :])             #Input: t --> t+seq_len
        targets.append(traj[:, i+1:i+seq_len+1, :])        #Output: t+1 --> t+seq_len+1
    
    #stack to make one tensor for lstm processing
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)

    return inputs, targets

class SpiralSequenceDataset(Dataset):
    def __init__(self, observed_data, full_data, observed_tp):
        self.observed_data = observed_data
        self.full_data = full_data
        self.observed_tp = observed_tp

        #Compute delta_t between consecutive observed timestamps
        delta_t = observed_tp[1:] - observed_tp[:-1]
        delta_t = torch.cat(
            [torch.zeros(1, dtype=observed_tp.dtype, device=observed_tp.device), delta_t], 
            dim=0
        )

        #Shape becomes [obs_len, 1] so it can be concatenated with (x, y)
        self.delta_t = delta_t.unsqueeze(-1)

    def __len__(self):
        return self.observed_data.size(0)
    
    def __getitem__(self, idx):
        obs = self.observed_data[idx]
        full_traj = self.full_data[idx]
        future = full_traj[obs.size(0):]

        #Append delta_t as a third feature to each observed point
        #LSTM input is now [x, y, delta_t]
        obs_with_dt = torch.cat([obs, self.delta_t], dim=-1)

        return obs_with_dt, future, full_traj
