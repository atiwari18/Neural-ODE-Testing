from models.latent_ode import generate_spiral2d
import torch
import numpy.random as npr
import numpy as np

data_rng = npr.RandomState(42)

latent_dim = 6
nhidden = 40
rnn_nhidden = 45
obs_dim = 2
nspiral = 1000
start = 0.
stop = 6 * np.pi
noise_std = 0.2
a = 0.
b = .3
ntotal = 1000
nsample = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# generate toy spiral data
orig_trajs, samp_trajs, orig_ts, samp_ts, t0_idxs = generate_spiral2d(
    nspiral=nspiral,
    start=start,
    stop=stop,
    noise_std=noise_std,
    a=a, b=b, savefig=False, rng=data_rng
)

# for i in range(5):
#     t0 = t0_idxs[i]
    
#     true_start_time = orig_ts[t0]
#     true_end_time = orig_ts[t0 + nsample - 1]
    
#     assumed_start_time = samp_ts[0]
#     assumed_end_time = samp_ts[-1]
    
#     print(f"Trajectory {i}:")
#     print(f"  TRUE time   = [{true_start_time:.3f}, {true_end_time:.3f}]")
#     print(f"  ASSUMED time= [{assumed_start_time:.3f}, {assumed_end_time:.3f}]")
#     print()

for i in range(5):
    t0 = t0_idxs[i]
    
    true_start_time = orig_ts[t0]
    true_end_time = orig_ts[t0 + nsample - 1]
    
    recovered_start_time = samp_ts[i][0]
    recovered_end_time = samp_ts[i][-1]
    
    print(f"Trajectory {i}:")
    print(f"  TRUE time       = [{true_start_time:.3f}, {true_end_time:.3f}]")
    print(f"  RECOVERED time  = [{recovered_start_time:.3f}, {recovered_end_time:.3f}]")
    print()