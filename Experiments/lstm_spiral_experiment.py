import argparse
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.generate_spirals import generate_spiral_extrap_dataset, load_or_create_shared_spiral_dataset
from models.lstm import Seq2SeqLSTM, plot_rollouts, split_train_test
from dataset.lstm_dataset import SpiralSequenceDataset

def parse_args():
    parser = argparse.ArgumentParser("LSTM spiral extrapolation")
    parser.add_argument("--nspiral", type=int, default=1000)
    parser.add_argument("--ntotal", type=int, default=500)
    parser.add_argument("--obs-len", type=int, default=40)
    parser.add_argument("--pred-len", type=int, default=160)
    parser.add_argument("--stop", type=float, default=18.85)
    parser.add_argument("--noise-std", type=float, default=0.1)

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher-forcing", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1991)
    parser.add_argument("--plot-every", type=int, default=25)
    parser.add_argument("--save-dir", type=str, default="LSTM_Spiral_Results")
    parser.add_argument("--shared_spiral_path", type=str, default="Experiments/shared_spiral_dataset.pt")
    parser.add_argument("--force_regen_shared", action="store_true")
    parser.add_argument("--irregular_spiral", action="store_true")
    parser.add_argument("--irregular_window_time", type=float, default=2 * np.pi)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

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
    irregular_window_time=args.irregular_window_time 
    )

    train_full, test_full, train_obs, test_obs = split_train_test(
        full_data, observed_data, train_frac=0.8
    )

    train_dataset = SpiralSequenceDataset(train_obs, train_full, observed_tp, observed_offsets)
    test_dataset = SpiralSequenceDataset(test_obs, test_full, observed_tp, observed_offsets)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=min(args.batch_size, len(test_dataset)), shuffle=False)

    model = Seq2SeqLSTM(
        input_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=2,
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

        for observed, future, full_traj in train_loader:
            observed = observed.to(device)
            future = future.to(device)

            optimizer.zero_grad()
            future_pred = model(
                observed,
                future_len=future.size(1),
                future_truth=future,
                teacher_forcing_ratio=args.teacher_forcing,
            )

            # print("observed:", observed.shape)
            # print("future:", future.shape)
            # print("future_pred:", future_pred.shape)


            loss = criterion(future_pred, future)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        test_losses = []

        with torch.no_grad():
            for observed, future, full_traj in test_loader:
                observed = observed.to(device)
                future = future.to(device)

                future_pred = model(
                    observed,
                    future_len=future.size(1),
                    future_truth=None,
                    teacher_forcing_ratio=0.0,
                )

                loss = criterion(future_pred, future)
                test_losses.append(loss.item())

        mean_train = float(np.mean(train_losses))
        mean_test = float(np.mean(test_losses))

        print(
            f"Epoch {epoch:04d} | "
            f"train_mse {mean_train:.6f} | "
            f"test_mse {mean_test:.6f}"
        )

        if mean_test < best_test:
            best_test = mean_test
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_test_mse": best_test,
                },
                os.path.join(args.save_dir, "best_lstm_spiral.pt"),
            )

        if epoch % args.plot_every == 0 or epoch == 1 or epoch == args.epochs:
            plot_rollouts(
                model=model,
                test_dataset=test_dataset,
                device=device,
                epoch=epoch,
                save_dir=os.path.join(args.save_dir, "plots"),
                plot_indices=[0, 1, 2, 3]
            )

