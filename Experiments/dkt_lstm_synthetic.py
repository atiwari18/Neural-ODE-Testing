import numpy as np
import argparse
import csv
import json
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from models.lstm import LSTM
from dataset.lstm_dataset import SyntheticKTDataset, split_train_val_test_syndkt

def parse_args():
    parser = argparse.ArgumentParser("LSTM Synthetic DKT")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)

    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=32)

    return parser.parse_args()

def save_training_plot(log_rows, save_dir):
    epochs = [row["epoch"] for row in log_rows]
    train_loss = [row["train_loss"] for row in log_rows]
    val_loss = [row["val_loss"] for row in log_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, linewidth=2, label="Train loss")
    plt.plot(epochs, val_loss, linewidth=2, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("LSTM DKT Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = save_dir / "training_validation_loss.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path

def save_metric_plot(log_rows, save_dir):
    epochs = [row["epoch"] for row in log_rows]
    train_auc = [row["train_auc"] for row in log_rows]
    val_auc = [row["val_auc"] for row in log_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_auc, linewidth=2, label="Train AUC")
    plt.plot(epochs, val_auc, linewidth=2, label="Validation AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("LSTM DKT AUC")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = save_dir / "training_validation_auc.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


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

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_count = 0
    
    all_probs = []
    all_targets = []

    for x, target_q, target_r in loader:
        x = x.to(device)                    # [batch, 49, 100]
        target_q = target_q.to(device)      # [batch, 49]
        target_r = target_r.to(device)

        output, _ = model(x)

        #Select prediction for the actual next question
        logits = output.gather(
            dim=2, 
            index=target_q.unsqueeze(-1)
        ).squeeze(-1)

        loss = criterion(logits, target_r)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
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

    accuracy = accuracy_score(all_targets, pred_labels)
    auc = roc_auc_score(all_targets, all_probs)

    return {
        "loss" : total_loss / total_count, 
        "accuracy" : accuracy, 
        "auc" : auc
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

    model = LSTM(
        input_dim=2 * dataset.num_questions,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=dataset.num_questions,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device         : {device}")
    print(f"Dataset        : {args.csv_path}")
    print(f"Save dir       : {save_dir}")
    print(f"Students       : {len(dataset)}")
    print(f"Questions      : {dataset.num_questions}")
    print(f"Train students : {len(train_dataset)}")
    print(f"Val students   : {len(val_dataset)}")
    print(f"Test students  : {len(test_dataset)}")
    print()

    log_rows = []

    best_val_loss = float("inf")
    best_epoch = None
    best_model_path = save_dir / "best_dkt_lstm_synthetic.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

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

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )

    final_model_path = save_dir / "final_dkt_lstm_synthetic.pt"
    torch.save(model.state_dict(), final_model_path)

    train_log_path = save_train_log(log_rows, save_dir)
    loss_plot_path = save_training_plot(log_rows, save_dir)
    auc_plot_path = save_metric_plot(log_rows, save_dir)

    best_row = log_rows[best_epoch - 1]

    final_metrics = {
        "csv_path": str(args.csv_path),
        "save_dir": str(save_dir),
        "seed": args.seed,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "lr": args.lr,

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