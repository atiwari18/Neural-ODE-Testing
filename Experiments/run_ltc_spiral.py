import argparse
import numpy as np
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.generate_spirals import load_or_create_shared_spiral_dataset
from models.ltc import Seq2SeqLTC, plot_ltc_rollouts
from dataset.ltc_dataset import SpiralLTCDataset, split_train_test

def parse_args():
    parser = argparse.ArgumentParser()

    #Dataset Settings
    parser.add_argument("--nspiral", type=int, default=1000)
    parser.add_argument("--ntotal", type=int, default=500)
    parser.add_argument("--obs-len", type=int, default=40)
    parser.add_argument("--pred_len", type=int, default=160)
    parser.add_argument("--stop", type=float, default=18.85)
    parser.add_argument("--noise-std", type=float, default=0.1)

    #Model Settings
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--mixed-memory", action="store_true")
    parser.add_argument("--ode-unfolds", type=int, default=6)

    #Training Settings
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-forcing", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1991)
    parser.add_argument("--plot-every", type=int, default=500)

    #Output Settings
    parser.add_argument("--save-dir", type=str, default="LTC_Results")
    parser.add_argument("--shared_spiral_path", type=str, default="Experiments/shared_spiral_dataset_scoring.pt")
    parser.add_argument("--force_regen_shared", action="store_true")
    parser.add_argument("--irregular_spiral", action="store_true")
    parser.add_argument("--irregular_window_time", type=float, default=2 * np.pi)
    parser.add_argument("--n_trials", type=int, default=100)

    #NCP settings
    parser.add_argument("--ncp", action="store_true")
    parser.add_argument("--sparsity_level", type=float, default=0.5)

    return parser.parse_args()

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_losses = []

    with torch.no_grad():
        for observed_xy, observed_dt, future_xy, future_dt, full_traj in test_loader:
            observed_xy = observed_xy.to(device)
            observed_dt = observed_dt.to(device)
            future_xy = future_xy.to(device)
            future_dt = future_dt.to(device)

            future_pred = model(
                observed_xy,
                observed_dt,
                future_dt,
                future_truth=None,
                teacher_forcing_ratio=0.0,
            )

            loss = criterion(future_pred, future_xy)
            test_losses.append(loss.item())

    return float(np.mean(test_losses))

if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    #Load or create dataset
    full_data, observed_data, full_tp, observed_tp, observed_offsets = load_or_create_shared_spiral_dataset(
        dataset_path=args.shared_spiral_path,
        nspiral=args.nspiral,
        ntotal=args.ntotal,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        start=0.0,
        stop=args.stop,
        noise_std=args.noise_std,
        a=0.0,
        b=0.3,
        savefig=True,
        device=device,
        force_regen=args.force_regen_shared,
        irregular=args.irregular_spiral,
        irregular_window_time=args.irregular_window_time,
        n_trials=args.n_trials,
    )

    #Generate Test/Train Split
    train_full, test_full, train_obs, test_obs = split_train_test(full_data, observed_data, train_frac=0.8)

    train_dataset = SpiralLTCDataset(train_obs, train_full, full_tp, observed_tp, observed_offsets)
    test_dataset = SpiralLTCDataset(test_obs, test_full, full_tp, observed_tp, observed_offsets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, len(test_dataset)), shuffle=False)

    if args.ncp:
        print("NCP!!!!")

    model = Seq2SeqLTC(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        output_dim=2,
        mixed_memory=args.mixed_memory,
        ode_unfolds=args.ode_unfolds,
        use_ncp=args.ncp, 
        sparsity_level=args.sparsity_level
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_test = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for observed_xy, observed_dt, future_xy, future_dt, full_traj in train_loader:
            observed_xy = observed_xy.to(device)
            observed_dt = observed_dt.to(device)
            future_xy = future_xy.to(device)
            future_dt = future_dt.to(device)

            optimizer.zero_grad()

            # print("observed_xy:", observed_xy.shape)
            # print("observed_dt:", observed_dt.shape)
            # print("future_xy:", future_xy.shape)
            # print("future_dt:", future_dt.shape)

            future_pred = model(
                observed_xy,
                observed_dt,
                future_dt,
                future_truth=future_xy,
                teacher_forcing_ratio=args.teacher_forcing,
            )

            loss = criterion(future_pred, future_xy)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        mean_train = float(np.mean(train_losses))
        mean_test = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch:04d} | "
            f"train_mse {mean_train:.6f} | "
            f"test_mse {mean_test:.6f}"
        )

        #Saving the best model
        if mean_test < best_test:
            best_test = mean_test
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_test_mse": best_test,
                },
                os.path.join(args.save_dir, "best_ltc_spiral.pt"),
            )

        #Plot every n epochs
        if epoch % args.plot_every == 0 or epoch == 1 or epoch == args.epochs:
            plot_ltc_rollouts(
                model=model,
                test_dataset=test_dataset,
                device=device,
                epoch=epoch,
                save_dir=os.path.join(args.save_dir, "plots"),
                plot_indices=[0, 1, 2, 3],
            )