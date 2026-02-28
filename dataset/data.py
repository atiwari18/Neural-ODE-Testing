import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from torchdiffeq import odeint_adjoint as odeint

#Sine dynamics
class SineDynamics(nn.Module):
    def __init__(self, device="cpu"):
        super(SineDynamics, self).__init__()
        self.A = torch.tensor([[0.0, 1.0], 
                          [-1.0, 0.0]], dtype=torch.float32, device=device)
        
    def forward(self, t, y):
        return torch.mm(y, self.A.T)

def generate_sine(true_func, batch_size=32, n_samples=100, t_max=4*np.pi, device="cpu"):
    #timepoints
    t = torch.linspace(0, t_max, n_samples).to(device)

    #Intitial conditions
    y0_position = (torch.rand(batch_size) * 4 - 2)
    y0_velocity = (torch.rand(batch_size) * 4 - 2)
    y0 = torch.stack([y0_position, y0_velocity], dim=1).to(device)

    #solve ode with true dynamics
    with torch.no_grad():
        trajectories = odeint(true_func, y0, t, method='dopri5')

    return t, y0, trajectories

#True sprial dynamics
class SpiralDynamics(nn.Module):
    def __init__(self, direction=1.0, device="cpu"):
        super(SpiralDynamics, self).__init__()

        #weight matrix for spiral
        self.A = torch.tensor([[-0.1, direction*-1.0], 
                               [direction*1.0, -0.1]], dtype=torch.float32, device=device)
        
    def forward(self, t, y):
        return torch.mm(y, self.A.T)

def generate_spiral(batch_size=32, n_samples=100, t_max=10, noise_std=0.0, device="cpu"):
    halb_batch = batch_size // 2
    #time
    t = torch.linspace(0, t_max, n_samples).to(device)
    
    #Initial conditions: random points on a circle
    theta = torch.rand(batch_size) * 2 * np.pi
    radius = 2.0
    y0 = torch.stack([
        radius * torch.cos(theta),
        radius * torch.sin(theta)
    ], dim=1).to(device)  # [batch_size, 2]
    
    #Solve ODE with true dynamics
    with torch.no_grad():
        #Clockwise
        true_func_cw = SpiralDynamics(direction=1.0, device=device)
        y0_cw = y0[:halb_batch]
        traj_cw = odeint(true_func_cw, y0_cw, t, method="dopri5")
        #Shape: [n_points, batch_size, 2]

        #Counter Clockwise
        true_func_ccw = SpiralDynamics(direction=-1.0, device=device)
        y0_ccw = y0[halb_batch:]
        traj_ccw = odeint(true_func_ccw, y0_ccw, t, method="dopri5")

        #Combine
        true_trajectories = torch.cat([traj_cw, traj_ccw], dim=1)

    #Add noise
    if noise_std > 0:
        true_trajectories += noise_std * torch.randn_like(true_trajectories)

    return t, y0, true_trajectories

def generate_irregular(subsample_points, trajectories, n_samples, t, batch_size, device):
    t = torch.zeros(subsample_points, dtype=torch.float32, device=device)
    subsampled_traj = torch.zeros(subsample_points, batch_size, 2, dtype=torch.float32, device=device)
    
    for b in range(batch_size):
        #random indicies, sort by time
        idx = torch.randperm(n_samples)[:subsample_points].sort()[0]
        subsampled_traj[:, b, :] = trajectories[idx, b, :]
        t = t[idx]

    return t, subsampled_traj 


def plot_samples(t, y0, trajectories, title="True Sine Wave Dynamics", file_name="true_sine.png"):
    # Move to CPU and convert to numpy
    t_np = t.cpu().numpy()
    traj_np = trajectories.cpu().numpy()  # [n_points, batch_size, 2]
    
    batch_size = traj_np.shape[1]
    n_plot = batch_size
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot position (y) vs time
    for i in range(n_plot):
        position = traj_np[:, i, 0]  # Extract position
        axes[0].plot(t_np, position, '-', alpha=0.7, linewidth=1.5, 
                    label=f'Trajectory {i+1}' if i < 5 else None)
    
    axes[0].set_title("Position vs Time", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Time (t)", fontsize=12)
    axes[0].set_ylabel("Position (y)", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot velocity (v) vs time
    for i in range(n_plot):
        velocity = traj_np[:, i, 1]  # Extract velocity
        axes[1].plot(t_np, velocity, '-', alpha=0.7, linewidth=1.5)
    
    axes[1].set_title("Velocity vs Time", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Time (t)", fontsize=12)
    axes[1].set_ylabel("Velocity (v)", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    full_path = os.path.join(results_dir, file_name)
    plt.savefig(full_path, dpi=150)
    plt.close()
    print(f"Plot saved to: {full_path}")

def plot_spiral(t, y0, trajectories, title="Spiral Trajectories", file_name="trajectories.png"):
    # Move to CPU and convert to numpy for plotting
    t_np = t.cpu().numpy()
    y0_np = y0.cpu().numpy()
    traj_np = trajectories.cpu().numpy()
    
    batch_size = y0_np.shape[0]
    
    plt.figure(figsize=(10, 10))
    
    # Plot each trajectory
    for i in range(batch_size):
        traj = traj_np[:, i, :]  # [n_points, 2]
        
        # Plot trajectory
        plt.plot(traj[:, 0], traj[:, 1], '-', alpha=0.6, linewidth=1.5)
        
        # Mark starting point
        plt.scatter(y0_np[i, 0], y0_np[i, 1], c='green', s=100, 
                   marker='o', zorder=5, edgecolors='black', linewidths=1)
        
        # Mark ending point
        plt.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, 
                   marker='x', zorder=5, linewidths=2)
    
    plt.title(title, fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add legend
    plt.scatter([], [], c='green', s=100, marker='o', 
               edgecolors='black', linewidths=1, label='Start')
    plt.scatter([], [], c='red', s=100, marker='x', 
               linewidths=2, label='End')
    plt.legend(loc='best')
    
    plt.tight_layout()

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    full_path = os.path.join(results_dir, file_name)

    # Save the figure
    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")

def generate_spiral2d(nspiral=1000, ntotal=500, nsample=100,
                      start=0., stop=6*np.pi, noise_std=0.3,
                      a=0., b=0.3, savefig=True):
    """Exact copy of the author's data generation (parametric spirals)."""
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # Clockwise
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    # Counter-clockwise
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
        print('Saved ground truth spiral at ./ground_truth.png')

    orig_trajs = []
    samp_trajs = []
    for _ in range(nspiral):
        t0_idx = np.random.multinomial(1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample
        cc = bool(np.random.rand() > .5)
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)
        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += np.random.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)
    return orig_trajs, samp_trajs, orig_ts, samp_ts


# Testing generation
if __name__ == '__main__':
    #t, state = generate_irregular(50)
    #plot_samples(t, state, title="Test", file_name="test.png")

    # true_func = SpiralDynamics()
    # t, y0, true_traj = generate_spiral(true_func, batch_size=8, n_samples=100)
    # plot_spiral(t, y0, true_traj, title="True Spiral Dynamics", file_name="true_spirals.png")

    true_func = SineDynamics()
    t, y0, true_traj = generate_sine(true_func, batch_size=16, n_samples=100)
    plot_spiral(t, y0, true_traj, title="Sine Phase Space", file_name="true_sine_phase.png")