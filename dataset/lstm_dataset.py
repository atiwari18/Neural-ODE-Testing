import torch

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
