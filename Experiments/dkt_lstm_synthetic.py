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
from dataset.lstm_dataset import SyntheticKTDataset, split_train_test_syndkt

def parse_args():
    parser = argparse.ArgumentParser("LSTM Synthetic DKT")
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)

    parser.add_argument("--train-size", type=int, default=2000)
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

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training BCE Loss")
    plt.title("LSTM DKT Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = save_dir / "training_loss.png"
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
    train_dataset, test_dataset = split_train_test_syndkt(dataset, train_size=args.train_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    print(f"Device: {device}")
    print(f"Dataset: {args.csv_path}")
    print(f"Students: {len(dataset)}")
    print(f"Questions: {dataset.num_questions}")
    print(f"Train students: {len(train_dataset)}")
    print(f"Test students: {len(test_dataset)}")

    log_rows = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        log_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_auc": train_metrics["auc"],
        }
        log_rows.append(log_row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_auc={train_metrics['auc']:.4f} | "
        )

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )

    model_path = save_dir / "dkt_lstm_synthetic.pt"
    torch.save(model.state_dict(), model_path)

    train_log_path = save_train_log(log_rows, save_dir)
    plot_path = save_training_plot(log_rows, save_dir)

    final_metrics = {
        "csv_path": str(args.csv_path),
        "save_dir": str(save_dir),
        "seed": args.seed,
        "train_size": args.train_size,
        "test_size": len(test_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "lr": args.lr,
        "final_train_loss": log_rows[-1]["train_loss"],
        "final_train_accuracy": log_rows[-1]["train_accuracy"],
        "final_train_auc": log_rows[-1]["train_auc"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auc": test_metrics["auc"],
        "model_path": str(model_path),
        "train_log_path": str(train_log_path),
        "training_loss_plot": str(plot_path),
    }
    
    metrics_path = save_dir / "final_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(final_metrics, f, indent=2)

    print()
    print(
        f"Final test | "
        f"test_loss={test_metrics['loss']:.4f} "
        f"test_acc={test_metrics['accuracy']:.4f} "
        f"test_auc={test_metrics['auc']:.4f}"
    )
    print(f"Saved model         : {model_path}")
    print(f"Saved train log     : {train_log_path}")
    print(f"Saved loss plot     : {plot_path}")
    print(f"Saved final metrics : {metrics_path}")