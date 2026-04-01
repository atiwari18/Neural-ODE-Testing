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
    def __init__(self, observed_data, full_data):
        self.observed_data = observed_data
        self.full_data = full_data

    def __len__(self):
        return self.observed_data.size(0)
    
    def __getitem__(self, idx):
        obs = self.observed_data[idx]
        full_traj = self.full_data[idx]
        future = full_traj[obs.size(0):]

        return obs, future, full_traj
