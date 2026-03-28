import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.ode_rnn import generate_spiral2d, LatentODE
import argparse
import numpy as np
from torchdiffeq import odeint

def parse_args():
    parser = argparse.ArgumentParser("Latent ODE-RNN Spiral Experiment")
    parser.add_argument('--niters',      type=int,   default=2000)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--latent_dim',  type=int,   default=6)
    parser.add_argument('--kl_coef',     type=float, default=1.0)
    parser.add_argument('--batch_size',  type=int,   default=128)
    parser.add_argument('--n_samples',   type=int,   default=1)   # IWAE samples
    parser.add_argument('--visualize',   action='store_true', default=True)
    parser.add_argument('--seed',        type=int,   default=42)
    return parser.parse_args()

def train(
    nspiral=500,
    ntotal=500,
    nsample=100,
    latent_dim=6,
    rec_dim=30,
    ode_hidden=64,
    gru_units=64,
    obsrv_std=0.3,
    lr=1e-3,
    niters=2000,
    batch_size=128,
    kl_coef=1.0,
    n_samples=1,
    log_every=100,
    start=0, 
    stop=6 * np.pi,
    device="cuda"
):
    print("=== Generating spiral data ===")
    orig_trajs, samp_trajs, orig_ts, samp_ts = generate_spiral2d(
        nspiral=nspiral, ntotal=ntotal, nsample=nsample, start=start, stop=stop,
        noise_std=0.3, savefig=True,
    )
 
    # Convert to tensors
    samp_trajs_t = torch.tensor(samp_trajs, dtype=torch.float32).to(device)  # (N, nsample, 2)
    samp_ts_t    = torch.tensor(samp_ts,    dtype=torch.float32).to(device)  # (nsample,)
    orig_trajs_t = torch.tensor(orig_trajs, dtype=torch.float32).to(device)  # (N, ntotal, 2)
    orig_ts_t    = torch.tensor(orig_ts,    dtype=torch.float32).to(device)  # (ntotal,)
 
    N = samp_trajs_t.shape[0]
 
    print(f"Dataset: {N} spirals, {nsample} observed points each")
    print(f"Model: latent_dim={latent_dim}, rec_dim={rec_dim}, ode_hidden={ode_hidden}")
 
    # Build model
    model = LatentODE(
        input_dim=2,
        latent_dim=latent_dim,
        rec_dim=rec_dim,
        ode_hidden=ode_hidden,
        gru_units=gru_units,
        obsrv_std=obsrv_std,
    ).to(device)
 
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=niters)
 
    train_losses = []
 
    print("\n=== Training ===")
    for itr in range(1, niters + 1):
        model.train()
        optimizer.zero_grad()
 
        # Random minibatch
        idx = torch.randperm(N)[:batch_size]
        batch = samp_trajs_t[idx]                          # (B, nsample, 2)
 
        loss, rec_ll, kl = model.compute_loss(
            batch, samp_ts_t,
            kl_coef=kl_coef,
            n_samples=n_samples,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
 
        train_losses.append(loss.item())
 
        if itr % log_every == 0 or itr == 1:
            print(
                f"  iter {itr:4d}/{niters} | loss={loss.item():.3f}"
                f" | rec_ll={rec_ll.item():.3f} | kl={kl.item():.3f}"
                f" | lr={scheduler.get_last_lr()[0]:.2e}"
            )
 
    print("\n=== Saving model ===")
    torch.save(model.state_dict(), "latent_ode_spiral.pt")
 
    #Visualise reconstructions
    print("=== Plotting desired visualization (2 CW + 2 CCW with forward/backward) ===")
    model.eval()
    with torch.no_grad():
        VIZ_SPIRAL_IDX = [0, 1, 2, 3]          # fixed indices like the paper/repo
 
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
 
        for plot_i, spiral_idx in enumerate(VIZ_SPIRAL_IDX):
            ax = axes[plot_i]
 
            # 1. Encode z0 for this single spiral
            single_data = samp_trajs_t[spiral_idx:spiral_idx+1]   # (1, nsample, 2)
            mask = torch.ones_like(single_data)
            data_w_mask = torch.cat([single_data, mask], dim=-1)
 
            mu_z0, sigma_z0 = model.encoder(data_w_mask, samp_ts_t)
            eps = torch.randn_like(mu_z0)
            z0 = mu_z0 + eps * sigma_z0                     # (1, 1, latent_dim)
            z0 = z0.squeeze(0).squeeze(0)                   # (latent_dim,)
 
            t0_i = samp_ts_t[0].item()
 
            # odeint(func, y0, t) returns shape (len(t), *y0.shape).
            # y0 = z0.unsqueeze(0) has shape (1, latent_dim), so odeint returns
            # (n_steps, 1, latent_dim).  We need to squeeze dim 1 (the batch dim)
            # before decoding, giving (n_steps, latent_dim) → decoder → (n_steps, 2).
 
            # 2. Forward solve: from the first observed time t0 to the end of the
            #    full original trajectory  — this is interpolation/reconstruction,
            #    matching the paper's Figure 2 (right side: ODE decode forward in time).
            ts_pos = torch.linspace(t0_i, orig_ts_t[-1].item(), 2000, device=device)
            sol_pos = odeint(model.ode_func, z0.unsqueeze(0), ts_pos)  # (n_steps, 1, latent_dim)
            xs_pos = model.decoder(sol_pos.squeeze(1)).cpu().numpy()   # (n_steps, 2)
 
            # 3. Backward solve: strictly *decreasing* t — torchdiffeq supports this
            #    natively (no manual flip needed).  This shows what the latent ODE
            #    predicts before the first observation, matching the paper's note that
            #    the model can answer non-standard queries such as predicting backwards
            #    in time (Section 3.2).  Guard against t0_i ≈ 0.
            if t0_i > 1e-6:
                ts_neg = torch.linspace(t0_i, 0.0, 2000, device=device)   # strictly decreasing
                sol_neg = odeint(model.ode_func, z0.unsqueeze(0), ts_neg)  # (n_steps, 1, latent_dim)
                xs_neg = model.decoder(sol_neg.squeeze(1)).cpu().numpy()   # (n_steps, 2)
            else:
                xs_neg = None
 
            # 4. Ground truth + observations
            orig_traj = orig_trajs_t[spiral_idx].cpu().numpy()
            samp_traj = samp_trajs_t[spiral_idx].cpu().numpy()
 
            ax.plot(orig_traj[:, 0], orig_traj[:, 1], 'g', label='true trajectory', linewidth=1.5)
            ax.plot(xs_pos[:, 0], xs_pos[:, 1], 'r', label='learned (t>0)', linewidth=1.2)
            if xs_neg is not None:
                ax.plot(xs_neg[:, 0], xs_neg[:, 1], 'c', label='learned (t<0)', linewidth=1.2)
            ax.scatter(samp_traj[:, 0], samp_traj[:, 1], label='sampled data', s=3, alpha=0.6)
 
            ax.set_title(f'Spiral {spiral_idx}')
            ax.legend(fontsize=7)
 
        plt.tight_layout()
        plt.savefig("reconstructions.png", dpi=300)
        plt.close()
        print("Saved reconstructions.png")
 
    #Plotting Training curve
    plt.figure(figsize=(6, 3))
    plt.plot(train_losses)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO loss")
    plt.title("Training curve")
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.close()
    print("Saved training_curve.png")
 
    return model, train_losses

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, losses = train(
        niters=args.niters,
        lr=args.lr,
        latent_dim=args.latent_dim,
        kl_coef=args.kl_coef,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        
        # keep all your other defaults (nspiral=500, etc.)
        nspiral=500,
        ntotal=500,
        nsample=100,
        rec_dim=30,
        ode_hidden=64,
        gru_units=64,
        obsrv_std=0.3,
        log_every=10,
        start=0, 
        stop=6 * np.pi,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Done.")
    if args.visualize:
        print("Visualizations saved (reconstructions.png + training_curve.png)")