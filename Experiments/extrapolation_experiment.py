from models.latent_ode import generate_spiral2d, LatentODEfunc, RecognitionRNN, Decoder, RunningAverageMeter, log_normal_pdf, normal_kl
import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint
import csv

def parse_args():
    parser = argparse.ArgumentParser("KL Annealing Experiment")

    #Arguments
    parser.add_argument('--visualize', type=eval, default=False)
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kl_anneal', type=eval, default=False, help='Use KL Annealing (TRUE) or fixed KL Weight (FALSE)')

    return parser.parse_args()

def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      nsample=100,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      savefig=True, 
                      rng=None):
    #If no rng is applied then fall back to the global state
    if rng is None:
        rng = np.random.mtrand._rand

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('./ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    start_idxs = []

    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        #independently draws a new random t0 for each of the 1000 spirals. 
        #So each trajectory gets its own random starting point, and the 100-point window 
        #[t0_idx : t0_idx + nsample] is cut from that unique position.
        t0_idx = rng.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample
        start_idxs.append(t0_idx)

        cc = bool(rng.rand() > .5)  # uniformly select rotation
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += rng.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)
    start_idxs = np.array(start_idxs)

    return orig_trajs, samp_trajs, orig_ts, samp_ts, start_idxs

#Indicies of the 4 spirals
VIZ_SPIRAL_IDX = [0, 1, 2, 3]

def visualize(func, rec, dec, orig_trajs, samp_trajs, samp_ts, orig_ts_np,
              start_idxs, latent_dim, device, save_path):
    with torch.no_grad():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for plot_i, spiral_idx in enumerate(VIZ_SPIRAL_IDX):
            ax = axes[plot_i]

            # encode exactly as training
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0_all = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            z0 = z0_all[spiral_idx]

            orig_traj = orig_trajs[spiral_idx].cpu().numpy()
            samp_traj = samp_trajs[spiral_idx].cpu().numpy()
            t0_idx = start_idxs[spiral_idx]
            abs_start_t = orig_ts_np[t0_idx]          # real start time for THIS spiral

            # Forward extrapolation
            ts_fwd = torch.linspace(abs_start_t, orig_ts_np[-1], 2000).float().to(device)
            xs_fwd = dec(odeint(func, z0, ts_fwd)).cpu().numpy()

            # Backward extrapolation (now trained!)
            back_span = abs_start_t - orig_ts_np[0]   # how far back we can go
            ts_back = torch.linspace(abs_start_t, abs_start_t - back_span, 1000).float().to(device)
            xs_back = dec(odeint(func, z0, ts_back)).cpu().numpy()

            ax.plot(orig_traj[:, 0], orig_traj[:, 1],
                    'g', label='true trajectory (full)', linewidth=1.5)
            ax.plot(xs_fwd[:, 0], xs_fwd[:, 1],
                    'r', label='learned forward', linewidth=1.2)
            ax.plot(xs_back[:, 0], xs_back[:, 1],
                    'c', label='learned backward', linewidth=1.2)
            ax.scatter(samp_traj[:, 0], samp_traj[:, 1],
                       label='observed data', s=4, alpha=0.7, color='blue')

            ax.set_title(f'Spiral {spiral_idx} (obs start at t≈{abs_start_t:.2f})')
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f'Saved bidirectional visualization at {save_path}')    
     

if __name__ == '__main__':
    args = parse_args()

    #Reproducible Dataset: Same Seed --> Same spirals for every run
    data_rng = npr.RandomState(args.seed)
    torch.manual_seed(args.seed)

    latent_dim = 4
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 2
    nspiral = 1000
    start = -3 * np.pi
    stop = 6 * np.pi
    noise_std = 0.2
    a = 0.
    b = .3
    ntotal = 1500
    nsample = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate toy spiral data
    orig_trajs, samp_trajs, orig_ts, samp_ts, start_idxs = generate_spiral2d(
        nspiral=nspiral,
        start=start,
        stop=stop,
        noise_std=noise_std,
        a=a, b=b, rng=data_rng
    )

    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)

    # model
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, nspiral).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()

    anneal_label = "kl_anneal" if args.kl_anneal else "no_kl_anneal"
    print(f'Starting training: niters={args.niters}, lr={args.lr}, 'f'kl_anneal={args.kl_anneal}, seed={args.seed}')
    
    log_path = f'./log-{args.niters}-{args.lr}-{anneal_label}.csv'
    log_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['iter', 'loss', 'elbo', 'kl', 'weighted_kl', 'kl_weight', 'extrap_mse'])

    try:
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            h = rec.initHidden().to(device)

            #Encode first half
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)

            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
            pred_x = dec(pred_z)

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            
            logpx = log_normal_pdf(
                samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                    pz0_mean, pz0_logvar).sum(-1)

            #Annealing strategy from: https://github.com/YuliaRubanova/latent_ode/blob/master/run_models.py#L259
            wait_until_kl_inc = args.niters // 10
            if args.kl_anneal:
                if itr < wait_until_kl_inc:
                    kl_weight = 0
                else:
                    kl_weight = (1 - 0.99 ** (itr - wait_until_kl_inc))
            else:
                kl_weight = 1.0

            loss = torch.mean(-logpx + kl_weight*analytic_kl, dim=0)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

            #raw per-iteration values
            loss_raw = loss.item()
            elbo_raw = logpx.mean().item()
            wkl_raw = kl_weight * analytic_kl.mean().item()

            csv_writer.writerow([itr, loss_raw, elbo_raw, analytic_kl.mean().item(), wkl_raw, kl_weight])

            print('Iter: {}, running avg elbo: {:.4f}, running kl: {:.4f}, kl_weight: {:.4f}'.format(itr, -loss_meter.avg, analytic_kl.mean().item(), kl_weight))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'orig_trajs': orig_trajs,
                'samp_trajs': samp_trajs,
                'orig_ts': orig_ts,
                'samp_ts': samp_ts,
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))

    log_file.close()
    print(f"Saved training log to {log_path}")
    print(f'Training complete after {itr} iters.')

    #ensures memory is returned to the driver cleanly.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.visualize:
        fig_name = f'./vis-{args.niters}-{args.lr}-{anneal_label}.png'
        visualize(
            func, rec, dec,
            orig_trajs, samp_trajs, samp_ts,
            orig_ts, start_idxs, latent_dim, device,
            save_path=fig_name
        )