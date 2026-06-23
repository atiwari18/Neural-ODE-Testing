import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

import lib.utils as utils
from dataset.lstm_dataset import SyntheticKTDataset, split_train_val_test_syndkt
from lib.diffeq_solver import DiffeqSolver
from lib.encoder_decoder import Encoder_z0_ODE_RNN
from lib.ode_func import ODEFunc

class ODE_RNN_DKT(nn.Module):
    def __init__(self, num_questions, latent_dim=64, ode_units=64, gru_units=64,
                 device=torch.device("cpu"), ode_method="euler"):
        super().__init__()

        self.num_questions = num_questions
        self.input_dim = 2 * num_questions
        self.latent_dim = latent_dim
        self.device = device

        ode_func_net = nn.Sequential(
            nn.Linear(latent_dim, ode_units), 
            nn.Tanh(), 
            nn.Linear(ode_units, ode_units), 
            nn.Tanh(), 
            nn.Linear(ode_units, latent_dim)
        )

        utils.init_network_weights(ode_func_net)

        ode_func = ODEFunc(
            input_dim = self.input_dim, 
            latent_dim=latent_dim, 
            ode_func_net=ode_func_net,
            device=device,
        )

        self.diffeq_solver = DiffeqSolver(
            input_dim=self.input_dim,
            ode_func=ode_func, 
            method=ode_method,
            latents=latent_dim, 
            odeint_rtol=1e-4,
            odeint_atol=1e-5,
            device=device
        )

        self.ode_rnn_encoder = Encoder_z0_ODE_RNN(
            latent_dim=latent_dim,
            input_dim=self.input_dim * 2,  # interaction vector + mask
            z0_diffeq_solver=self.diffeq_solver,
            n_gru_units=gru_units,
            device=device,
        )

        self.output_layer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, num_questions),
        )

        utils.init_network_weights(self.output_layer)

    def forward(self, x, time_steps):
        # x shape: [batch, seq_len, 2 * num_questions]
        # mask shape matches x; synthetic data has every interaction observed.
        mask = torch.ones_like(x)
        x_and_mask = torch.cat([x, mask], dim=-1)

        _, _, latent_ys, _ = self.ode_rnn_encoder.run_odernn(
            x_and_mask,
            time_steps,
            run_backwards=False,
        )

        # run_odernn returns [1, seq_len, batch, latent_dim]
        latent_ys = latent_ys.permute(0, 2, 1, 3).squeeze(0)
        # shape: [batch, seq_len, latent_dim]

        logits = self.output_layer(latent_ys)
        # shape: [batch, seq_len, num_questions]

        return logits
    
def parse_args():
    parser = argparse.ArgumentParser("ODE-RNN Synthetic DKT")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)

    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--val-size", type=int, default=400)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--ode-units", type=int, default=64)
    parser.add_argument("--gru-units", type=int, default=64)
    parser.add_argument("--ode-method", type=str, default="euler")

    parser.add_argument("--seed", type=int, default=32)

    return parser.parse_args()

def save_train_log(log_rows, save_dir):
    log_path = save_dir / "train_log.csv"

    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_auc",
                "val_loss",
                "val_accuracy",
                "val_auc",
            ],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    return log_path


def save_loss_plot(log_rows, save_dir):
    epochs = [row["epoch"] for row in log_rows]
    train_loss = [row["train_loss"] for row in log_rows]
    val_loss = [row["val_loss"] for row in log_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, linewidth=2, label="Train loss")
    plt.plot(epochs, val_loss, linewidth=2, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("ODE-RNN DKT Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = save_dir / "training_validation_loss.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


def save_auc_plot(log_rows, save_dir):
    epochs = [row["epoch"] for row in log_rows]
    train_auc = [row["train_auc"] for row in log_rows]
    val_auc = [row["val_auc"] for row in log_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_auc, linewidth=2, label="Train AUC")
    plt.plot(epochs, val_auc, linewidth=2, label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("ODE-RNN DKT AUC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = save_dir / "training_validation_auc.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path

def run_epoch(model, loader, criterion, device, time_steps, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_count = 0

    all_probs = []
    all_targets = []

    for x, target_q, target_r in loader:
        x = x.float().to(device)
        target_q = target_q.long().to(device)
        target_r = target_r.float().to(device)

        output = model(x, time_steps)

        logits = output.gather(
            dim=2,
            index=target_q.unsqueeze(-1),
        ).squeeze(-1)

        loss = criterion(logits, target_r)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        count = target_r.numel()
        total_loss += loss.item() * count
        total_count += count

        probs = torch.sigmoid(logits).detach().cpu().reshape(-1).numpy()
        targets = target_r.detach().cpu().reshape(-1).numpy()

        all_probs.append(probs)
        all_targets.append(targets)

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    pred_labels = (all_probs >= 0.5).astype(np.float32)

    return {
        "loss": total_loss / total_count,
        "accuracy": accuracy_score(all_targets, pred_labels),
        "auc": roc_auc_score(all_targets, all_probs),
    }

if __name__ == '__main__':
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SyntheticKTDataset(args.csv_path)

    train_dataset, val_dataset, test_dataset = split_train_val_test_syndkt(
        dataset,
        train_size=args.train_size,
        val_size=args.val_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ODE_RNN_DKT(
        num_questions=dataset.num_questions,
        latent_dim=args.latent_dim,
        ode_units=args.ode_units,
        gru_units=args.gru_units,
        device=device,
        ode_method=args.ode_method,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #The synthetic KT data has no real timestamps, so we use question index time.
    #Inputs have length 49 because q0..q48 predict q1..q49.
    time_steps = torch.arange(
        dataset.num_questions - 1,
        dtype=torch.float32,
        device=device,
    )

    print(f"Device         : {device}")
    print(f"Dataset        : {args.csv_path}")
    print(f"Save dir       : {save_dir}")
    print(f"Students       : {len(dataset)}")
    print(f"Questions      : {dataset.num_questions}")
    print(f"Train students : {len(train_dataset)}")
    print(f"Val students   : {len(val_dataset)}")
    print(f"Test students  : {len(test_dataset)}")
    print(f"Latent dim     : {args.latent_dim}")
    print(f"ODE units      : {args.ode_units}")
    print(f"GRU units      : {args.gru_units}")
    print(f"ODE method     : {args.ode_method}")
    print()

    log_rows = []

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    min_delta = 1e-4

    best_model_path = save_dir / "best_dkt_ode_rnn_synthetic.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            time_steps=time_steps,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            time_steps=time_steps,
            optimizer=None,
        )

        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        log_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
        }
        log_rows.append(log_row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_auc={train_metrics['auc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f}"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch was {best_epoch} with val_loss={best_val_loss:.4f}."
            )
            
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        time_steps=time_steps,
        optimizer=None,
    )

    final_model_path = save_dir / "final_dkt_ode_rnn_synthetic.pt"
    torch.save(model.state_dict(), final_model_path)

    train_log_path = save_train_log(log_rows, save_dir)
    loss_plot_path = save_loss_plot(log_rows, save_dir)
    auc_plot_path = save_auc_plot(log_rows, save_dir)

    best_row = log_rows[best_epoch - 1]

    final_metrics = {
        "csv_path": str(args.csv_path),
        "save_dir": str(save_dir),
        "seed": args.seed,

        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),

        "epochs": args.epochs,
        "epochs_completed": log_rows[-1]["epoch"],
        "patience": args.patience,
        "best_epoch": best_epoch,

        "batch_size": args.batch_size,
        "lr": args.lr,
        "latent_dim": args.latent_dim,
        "ode_units": args.ode_units,
        "gru_units": args.gru_units,
        "ode_method": args.ode_method,

        "best_train_loss": best_row["train_loss"],
        "best_train_accuracy": best_row["train_accuracy"],
        "best_train_auc": best_row["train_auc"],

        "best_val_loss": best_row["val_loss"],
        "best_val_accuracy": best_row["val_accuracy"],
        "best_val_auc": best_row["val_auc"],

        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],

        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "train_log_path": str(train_log_path),
        "loss_plot_path": str(loss_plot_path),
        "auc_plot_path": str(auc_plot_path),
    }

    metrics_path = save_dir / "final_metrics.json"

    with metrics_path.open("w") as f:
        json.dump(final_metrics, f, indent=2)

    print()
    print(f"Best epoch by validation loss: {best_epoch}")
    print(
        f"Best val | "
        f"val_loss={best_row['val_loss']:.4f} "
        f"val_acc={best_row['val_accuracy']:.4f} "
        f"val_auc={best_row['val_auc']:.4f}"
    )
    
    print(
        f"Final test from best checkpoint | "
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_auc={test_metrics['auc']:.4f}"
    )

    print(f"Saved best model     : {best_model_path}")
    print(f"Saved final model    : {final_model_path}")
    print(f"Saved train log      : {train_log_path}")
    print(f"Saved loss plot      : {loss_plot_path}")
    print(f"Saved AUC plot       : {auc_plot_path}")
    print(f"Saved final metrics  : {metrics_path}")



