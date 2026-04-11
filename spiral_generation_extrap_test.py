import torch
import numpy as np
from lib.generate_spirals import generate_spiral_extrap_dataset, plot_spiral_dataset_example

device = torch.device("cpu")

full_data, obs_data, full_tp, obs_tp, offsets = generate_spiral_extrap_dataset(
    nspiral=10,
    ntotal=1000,
    obs_len=15,
    pred_len=800,
    start=0.0,
    stop=6 * torch.pi,
    noise_std=0.1,
    a=0.0,
    b=0.3,
    savefig=True,
    device=device,
    irregular=True, 
    irregular_window_time=2 * torch.pi,
)

plot_spiral_dataset_example(full_data, obs_data, full_tp, obs_tp, idx=3, savepath="spiral_dataset_example_scoring.png")
