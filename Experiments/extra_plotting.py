import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.distributions.normal import Normal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.lstm_dataset import SpiralSequenceDataset
from lib.generate_spirals import load_or_create_shared_spiral_dataset
from models.lstm import Seq2SeqLSTM, plot_rollouts, split_train_val_test


def parse_args():
    parser = argparse.ArgumentParser("Plot extra spiral rollouts for LSTM or ODE-RNN")

    parser.add_argument(
        "--model",
        choices=["lstm", "ode-rnn"],
        required=True,
        help="Model type to plot.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best_lstm_spiral.pt or experiment_XXXXX.ckpt.",
    )

    parser.add_argument(
        "--shared-spiral-path",
        default="Experiments/shared_spiral_dataset_scoring_pi.pt",
        help="Path to an existing shared spiral dataset .pt file.",
    )

    parser.add_argument(
        "--save-dir",
        default="extra_spiral_plots",
        help="Directory where the plot image will be saved.",
    )

    parser.add_argument(
        "--n-plots",
        type=int,
        default=16,
        help="Number of spiral trajectories to plot.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First test trajectory index to plot.",
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=1991,
        help="Random seed.",
    )

    return parser.parse_args()


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def as_namespace(args_obj):
    if isinstance(args_obj, dict):
        return SimpleNamespace(**args_obj)
    return args_obj

def get_plot_indices(args):
    return list(range(args.start_index, args.start_index + args.n_plots))


def build_lstm_test_dataset(lstm_args, device, args):
    full_data, observed_data, full_tp, observed_tp, observed_offsets = load_or_create_shared_spiral_dataset(
        dataset_path=args.shared_spiral_path,
        nspiral=getattr(lstm_args, "nspiral", 1000),
        ntotal=getattr(lstm_args, "ntotal", 1000),
        obs_len=getattr(lstm_args, "obs_len", 15),
        pred_len=getattr(lstm_args, "pred_len", 800),
        start=0.0,
        stop=getattr(lstm_args, "stop", 18.85),
        noise_std=getattr(lstm_args, "noise_std", 0.1),
        a=0.0,
        b=0.3,
        savefig=False,
        device=device,
        force_regen=False,
        irregular=getattr(lstm_args, "irregular_spiral", False),
        irregular_window_time=getattr(lstm_args, "irregular_window_time", np.pi),
        n_trials=getattr(lstm_args, "n_trials", 100),
    )

    train_full, val_full, test_full, train_obs, val_obs, test_obs = split_train_val_test(
        full_data,
        observed_data,
        train_frac=0.7,
        val_frac=0.15,
    )

    return SpiralSequenceDataset(test_obs, test_full, observed_tp, observed_offsets)


def plot_lstm(args, device, plot_indices):
    checkpoint = load_checkpoint(args.checkpoint, device)
    lstm_args = as_namespace(checkpoint["args"])

    model = Seq2SeqLSTM(
        input_dim=3,
        hidden_dim=lstm_args.hidden_dim,
        num_layers=lstm_args.num_layers,
        dropout=getattr(lstm_args, "dropout", 0.0),
        output_dim=2,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = build_lstm_test_dataset(lstm_args, device, args)

    epoch = int(checkpoint.get("epoch", getattr(lstm_args, "epochs", 0)))

    plot_rollouts(
        model=model,
        test_dataset=test_dataset,
        device=device,
        epoch=epoch,
        save_dir=args.save_dir,
        plot_indices=plot_indices,
    )


def import_ode_rnn_plotting_tools():
    old_argv = sys.argv[:]

    try:
        sys.argv = ["run_models.py"]

        from Experiments.run_models import plot_spiral_extrapolation
        from lib.create_latent_ode_model import create_LatentODE_model
        from lib.parse_datasets import parse_datasets
        import lib.utils as utils

        return plot_spiral_extrapolation, create_LatentODE_model, parse_datasets, utils

    finally:
        sys.argv = old_argv


def plot_ode_rnn(args, device, plot_indices):
    plot_spiral_extrapolation, create_LatentODE_model, parse_datasets, utils = import_ode_rnn_plotting_tools()

    checkpoint = load_checkpoint(args.checkpoint, device)
    ode_args = as_namespace(checkpoint["args"])
    ode_args.shared_spiral_path = args.shared_spiral_path

    data_obj = parse_datasets(ode_args, device)
    input_dim = data_obj["input_dim"]

    obsrv_std = torch.Tensor([0.01]).to(device)
    z0_prior = Normal(
        torch.Tensor([0.0]).to(device),
        torch.Tensor([1.0]).to(device),
    )

    model = create_LatentODE_model(
        ode_args,
        input_dim,
        z0_prior,
        obsrv_std,
        device,
        classif_per_tp=data_obj.get("classif_per_tp", False),
        n_labels=data_obj.get("n_labels", 1),
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_dict = utils.get_next_batch(data_obj["test_dataloader"])

    epoch = int(getattr(ode_args, "niters", 0))
    experiment_id = Path(args.checkpoint).stem.replace("experiment_", "")

    plot_spiral_extrapolation(
        test_dict=test_dict,
        model=model,
        epoch=epoch,
        experimentID=experiment_id,
        save_dir=args.save_dir,
        plot_indices=plot_indices,
    )


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_indices = get_plot_indices(args)

    if args.model == "lstm":
        plot_lstm(args, device, plot_indices)

    elif args.model == "ode-rnn":
        plot_ode_rnn(args, device, plot_indices)


if __name__ == "__main__":
    main()