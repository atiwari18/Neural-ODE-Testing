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

#Indicies of the 4 spirals
VIZ_SPIRAL_IDX = [0, 1, 2, 3]

def visualize(func, rec, dec, orig_trajs, samp_trajs, samp_ts, orig_ts_np, latent_dim, device, save_path):
    with torch.no_grad():
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for plot_i, spiral_idx in enumerate(VIZ_SPIRAL_IDX):
            ax = axes[plot_i]

            # --- encode the full batch, then pick the desired spiral ---
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0_all = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            z0 = z0_all[spiral_idx]  # shape: (latent_dim,)

            ts_pos = torch.from_numpy(
                np.linspace(0., 2. * np.pi, num=2000)
            ).float().to(device)
            ts_neg = torch.from_numpy(
                np.linspace(-np.pi, 0., num=2000)[::-1].copy()
            ).float().to(device)

            xs_pos = dec(odeint(func, z0, ts_pos)).cpu().numpy()
            xs_neg = torch.flip(
                dec(odeint(func, z0, ts_neg)), dims=[0]
            ).cpu().numpy()

            orig_traj = orig_trajs[spiral_idx].cpu().numpy()
            samp_traj = samp_trajs[spiral_idx].cpu().numpy()

            ax.plot(orig_traj[:, 0], orig_traj[:, 1],
                    'g', label='true trajectory', linewidth=1.5)
            ax.plot(xs_pos[:, 0], xs_pos[:, 1],
                    'r', label='learned (t>0)', linewidth=1.2)
            ax.plot(xs_neg[:, 0], xs_neg[:, 1],
                    'c', label='learned (t<0)', linewidth=1.2)
            ax.scatter(samp_traj[:, 0], samp_traj[:, 1],
                       label='sampled data', s=3, alpha=0.6)

            ax.set_title(f'Spiral {spiral_idx}')
            ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f'Saved visualization figure at {save_path}')       

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
    start = 0.
    stop = 6 * np.pi
    noise_std = 0.2
    a = 0.
    b = .3
    ntotal = 1000
    nsample = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate toy spiral data
    orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
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
    csv_writer.writerow(['iter', 'loss', 'elbo', 'kl', 'weighted_kl', 'kl_weight'])

    try:
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            h = rec.initHidden().to(device)
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
            orig_ts, latent_dim, device,
            save_path=fig_name
        )