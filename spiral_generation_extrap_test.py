import torch
from lib.generate_spirals import generate_spiral_extrap_dataset, plot_spiral_dataset_example

device = torch.device("cpu")

full_data, obs_data, full_tp, obs_tp = generate_spiral_extrap_dataset(
    nspiral=10,
    ntotal=500,
    obs_len=40,
    pred_len=160,
    start=0.0,
    stop=6 * torch.pi,
    noise_std=0.1,
    a=0.0,
    b=0.3,
    savefig=True,
    device=device,
)

plot_spiral_dataset_example(full_data, obs_data, full_tp, obs_tp, idx=3, savepath="spiral_dataset_example.png")
