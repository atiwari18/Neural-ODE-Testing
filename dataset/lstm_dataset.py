import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import torch.nn.functional as F

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
    def __init__(self, observed_data, full_data, observed_tp, observed_offsets):
        self.observed_data = observed_data
        self.full_data = full_data
        self.observed_tp = observed_tp
        self.observed_offsets = observed_offsets

        #Compute delta_t between consecutive observed timestamps
        delta_t = observed_tp[1:] - observed_tp[:-1]
        delta_t = torch.cat(
            [torch.zeros(1, dtype=observed_tp.dtype, device=observed_tp.device), delta_t], 
            dim=0
        )

        #Shape becomes [obs_len, 1] so it can be concatenated with (x, y)
        self.delta_t = delta_t.unsqueeze(-1)

        #The future should start after the last observed dense index
        self.last_obs_dense_idx = int(observed_offsets[-1].item())

    def __len__(self):
        return self.observed_data.size(0)
    
    def __getitem__(self, idx):
        obs = self.observed_data[idx]
        full_traj = self.full_data[idx]
        #future = full_traj[obs.size(0):]

        #Append delta_t as a third feature to each observed point
        #LSTM input is now [x, y, delta_t]
        obs_with_dt = torch.cat([obs, self.delta_t], dim=-1)

        #Start the future target immediately after the last observed dense point.
        future = full_traj[self.last_obs_dense_idx + 1:]

        return obs_with_dt, future, full_traj
    
class SyntheticKTDataset(Dataset):
    def __init__(self, csv_path):
        responses = np.loadtxt(csv_path, delimiter=",", dtype=np.int64)

        if responses.ndim == 1:
            responses = responses[None, :]

        self.responses = torch.tensor(responses, dtype=torch.long)
        self.num_students, self.num_questions = self.responses.shape

    def __len__(self):
        return self.num_students
    
    def __getitem__(self, index):
        responses = self.responses[index]
        question_ids = torch.arange(self.num_questions, dtype=torch.long)

        input_q = question_ids[:-1]
        input_r = responses[:-1]

        target_q = question_ids[1:]
        target_r = responses[1:].float()

        #Classic DKT encoding
        interaction_ids = input_q + input_r * self.num_questions

        x = F.one_hot(interaction_ids, num_classes=2 * self.num_questions).float()

        return x, target_q, target_r
    
def split_train_val_test_syndkt(dataset, train_size=1600, val_size=400):
    train_start = 0
    train_end = train_size

    val_start = train_end
    val_end = train_end + val_size

    test_start = val_end
    test_end = len(dataset)

    train_indices = list(range(train_start, train_end))
    val_indices = list(range(val_start, val_end))
    test_indices = list(range(test_start, test_end))

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset
