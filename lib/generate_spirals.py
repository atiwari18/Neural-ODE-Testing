import numpy as np
import numpy.random as npr
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def irregularity_score(x):
    """
    Score irregularity by the std of consecutive gaps.
    Higher means more uneven spacing.
    """
    x = np.asarray(x)

    #If a single vector is passed in, convert it to shape [1, n_points]
    if x.ndim == 1:
        x = x.reshape(1, -1)

    xs = np.sort(x, axis=1)
    d = xs[:, 1:] - xs[:, :-1]
    return np.sort(d, axis=1)

def choose_best_offset(candidate_idx, obs_len, full_local_ts, n_trials=50):
    """
    Sample several candidate offset sets and keep the one with
    the highest irregularity score.
    """
    best_offsets = None
    best_score = -np.inf

    #Make sure the array is a flat array
    full_local_ts = np.asarray(full_local_ts).reshape(-1)

    for _ in range(n_trials):
        offsets = np.sort(npr.choice(candidate_idx, size=obs_len, replace=False))
        
        #Pull the sample times out for this one candidate.
        trial_times = full_local_ts[offsets]

        #Score the spacing pattern itself. 
        score = float(np.ravel(irregularity_score(trial_times))[0])

        if score > best_score:
            best_score = score
            best_offsets = offsets

    return best_offsets, best_score

def load_or_create_shared_spiral_dataset(
    dataset_path,
    nspiral=1000,
    ntotal=1000,
    obs_len=40,
    pred_len=200,
    start=0.0,
    stop=6 * np.pi,
    noise_std=0.1,
    a=0.0,
    b=0.3,
    savefig=False,
    device=torch.device("cpu"),
    force_regen=False,
    irregular=False, 
    irregular_window_time=2
):
    #Convert to a Path object so path handling is easier and more reliable.
    dataset_path = Path(dataset_path)

    # Record the exact dataset settings so we can detect accidental mismatches.
    expected_config = {
        "nspiral": int(nspiral),
        "ntotal": int(ntotal),
        "obs_len": int(obs_len),
        "pred_len": int(pred_len),
        "start": float(start),
        "stop": float(stop),
        "noise_std": float(noise_std),
        "a": float(a),
        "b": float(b),
        "irregular" : bool(irregular), 
        "irregular_window_time" : float(irregular_window_time)
    }

    # If the file already exists and we are not forcing regeneration,
    # load it and verify that its settings match the current request.
    if dataset_path.exists() and not force_regen:
        saved = torch.load(dataset_path, map_location="cpu", weights_only=False)
        saved_config = saved.get("config", {})

        # Refuse to silently use the wrong dataset.
        if saved_config != expected_config:
            raise ValueError(
                f"Shared spiral dataset config mismatch at {dataset_path}. "
                f"Expected {expected_config}, found {saved_config}. "
                "Delete the file, choose a different path, or regenerate it with force_regen=True."
            )

        # Move the saved tensors to the requested device before returning them.
        return (
            saved["full_data"].to(device),
            saved["observed_data"].to(device),
            saved["full_time_steps"].to(device),
            saved["observed_time_steps"].to(device),
            saved["observed_offsets_t"].to(device)
        )

    # If the dataset file does not exist yet, generate it now.
    full_data, observed_data, full_time_steps, observed_time_steps, observed_offsets_t = generate_spiral_extrap_dataset(
        nspiral=nspiral,
        ntotal=ntotal,
        obs_len=obs_len,
        pred_len=pred_len,
        start=start,
        stop=stop,
        noise_std=noise_std,
        a=a,
        b=b,
        savefig=savefig,
        device=device,
        irregular=irregular, 
        irregular_window_time=irregular_window_time
    )

    # Make sure the parent folder exists before saving.
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CPU copies so the file can be loaded on any machine/device later.
    torch.save(
        {
            "config": expected_config,
            "full_data": full_data.detach().cpu(),
            "observed_data": observed_data.detach().cpu(),
            "full_time_steps": full_time_steps.detach().cpu(),
            "observed_time_steps": observed_time_steps.detach().cpu(),
            "observed_offsets_t" : observed_offsets_t.detach().cpu()
        },
        dataset_path,
    )

    return full_data, observed_data, full_time_steps, observed_time_steps, observed_offsets_t


def plot_spiral_dataset_example(full_data, observed_data, full_tp, observed_tp, idx=0, savepath=None):
    full_traj = full_data[idx].detach().cpu().numpy()
    obs_traj = observed_data[idx].detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.plot(full_traj[:, 0], full_traj[:, 1], color="lightgray", linewidth=2, label="target trajectory")
    plt.plot(obs_traj[:, 0], obs_traj[:, 1], "bo-", markersize=3, linewidth=1, label="observed prefix")
    plt.scatter(full_traj[0, 0], full_traj[0, 1], color="green", s=50, label="start")
    plt.scatter(full_traj[-1, 0], full_traj[-1, 1], color="red", s=50, label="end")
    plt.axis("equal")
    plt.legend()
    plt.title("Spiral dataset example")

    if savepath is not None:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    else:
        plt.show()

def _make_base_spirals(ntotal=1000, start=0.0, stop=6 * np.pi, a=0.0, b=0.3):
    """
    Build two full reference spirals on a common global timeline:
    - clockwise
    - counter-clockwise

    Returns:
        traj_cw: [ntotal, 2]
        traj_ccw: [ntotal, 2]
        ts: [ntotal]
    """
    ts = np.linspace(start, stop, num=ntotal)

    # Clockwise spiral
    zs_cw = stop + 1.0 - ts
    rs_cw = a + b * 50.0 / zs_cw
    xs_cw = rs_cw * np.cos(zs_cw) - 5.0
    ys_cw = rs_cw * np.sin(zs_cw)
    traj_cw = np.stack((xs_cw, ys_cw), axis=1)

    # Counter-clockwise spiral
    zs_cc = ts
    rs_cc = a + b * zs_cc
    xs_cc = rs_cc * np.cos(zs_cc) + 5.0
    ys_cc = rs_cc * np.sin(zs_cc)
    traj_cc = np.stack((xs_cc, ys_cc), axis=1)

    return traj_cw, traj_cc, ts


def generate_spiral_extrap_dataset(
    nspiral=1000,
    ntotal=1000,
    obs_len=40,
    pred_len=200,
    start=0.0,
    stop=6 * np.pi,
    noise_std=0.1,
    a=0.0,
    b=0.3,
    savefig=False,
    device=torch.device("cpu"),
    irregular=False,
    irregular_window_time=2 * np.pi,
):
    """
    Create a dataset for long-horizon extrapolation.

    Each sample is built by:
    1. choosing one of the two spiral families
    2. choosing a random start index t0_idx
    3. extracting a window of length pred_len
    4. defining local timestamps for that window
    5. adding noise only to the observed prefix [0:obs_len]

    Returns:
        full_windows: [nspiral, pred_len, 2]
            The target trajectories the model should reconstruct/predict.
        observed_windows: [nspiral, obs_len, 2]
            The prefix actually shown to the model.
        full_time_steps: [pred_len]
            Local time for the full prediction horizon.
        observed_time_steps: [obs_len]
            Local time for the observed prefix.
    """
    if obs_len >= pred_len:
        raise ValueError("obs_len must be smaller than pred_len for extrapolation.")

    if pred_len >= ntotal:
        raise ValueError("pred_len must be smaller than ntotal.")

    traj_cw, traj_cc, global_ts = _make_base_spirals(
        ntotal=ntotal, start=start, stop=stop, a=a, b=b
    )

    if savefig:
        plt.figure()
        plt.plot(traj_cw[:, 0], traj_cw[:, 1], label="clockwise")
        plt.plot(traj_cc[:, 0], traj_cc[:, 1], label="counter-clockwise")
        plt.legend()
        plt.savefig("./ground_truth_spirals.png", dpi=300)

    max_start = ntotal - pred_len
    if max_start <= 0:
        raise ValueError("ntotal must be larger than pred_len.")
    
    # Full target is always on a dense local time grid.
    full_local_ts = global_ts[:pred_len] - global_ts[0]

    if irregular:
        # Pick candidate dense-grid points whose local time is within the
        # requested observation window, e.g. [0, 2*pi].
        candidate_idx = np.where(full_local_ts <= irregular_window_time)[0]

        if len(candidate_idx) < obs_len:
            raise ValueError(
                f"irregular_window_time={irregular_window_time} only contains "
                f"{len(candidate_idx)} available dense points, but obs_len={obs_len}. "
                "Increase irregular_window_time, increase ntotal, or reduce obs_len."
            )

        if candidate_idx[-1] >= pred_len - 1:
            raise ValueError(
                "irregular_window_time reaches the end of the prediction horizon. "
                "Reduce irregular_window_time or increase pred_len so there is future left to extrapolate."
            )

        # # Choose obs_len irregularly spaced observations from that early-time region.
        # observed_offsets = np.sort(
        #     npr.choice(candidate_idx, size=obs_len, replace=False)
        # )

        #Try several candidate irregular patterns and keep the most irregular one. 
        observed_offsets, best_score = choose_best_offset(candidate_idx=candidate_idx, 
                                                          obs_len=obs_len, 
                                                          full_local_ts=full_local_ts, 
                                                          n_trials=100)

        print(f"Selected shared irregular offsets with score {best_score:.6f}")

    else:
        # Regular observed prefix.
        observed_offsets = np.arange(obs_len)

    observed_local_ts = full_local_ts[observed_offsets]

    full_windows = []
    observed_windows = []

    for _ in range(nspiral):
        # Randomly choose which spiral family this sample comes from.
        use_cc = bool(npr.rand() > 0.5)
        source_traj = traj_cc if use_cc else traj_cw

        # Choose a random valid start so we can still take pred_len points.
        t0_idx = npr.randint(0, max_start)

        # Full target trajectory remains dense.
        full_window = source_traj[t0_idx : t0_idx + pred_len].copy()

        # Observed points come from the chosen offsets.
        observed_window = full_window[observed_offsets].copy()

        # Add noise only to observed points.
        observed_window += npr.randn(*observed_window.shape) * noise_std

        full_windows.append(full_window)
        observed_windows.append(observed_window)

    full_windows = torch.tensor(np.stack(full_windows, axis=0), dtype=torch.float32, device=device)
    observed_windows = torch.tensor(np.stack(observed_windows, axis=0), dtype=torch.float32, device=device)

    full_time_steps = torch.tensor(full_local_ts, dtype=torch.float32, device=device)
    observed_time_steps = torch.tensor(observed_local_ts, dtype=torch.float32, device=device)

    #Return the dense offsets used to choose irregular observed points
    #This is needed by the LSTM dataset so it know where the observed region ends.
    observed_offsets_t = torch.tensor(observed_offsets, dtype=torch.long, device=device)

    return full_windows, observed_windows, full_time_steps, observed_time_steps, observed_offsets_t
