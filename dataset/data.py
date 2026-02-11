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

def generate_irregular(true_func, n_samples=100, t_max=4*np.pi, noise_std=0.1, device="cpu"):
    t = np.sort(np.random.rand(n_samples) * t_max)
    t[0] = 0
    t_tensor = torch.tensor(t, dtype=torch.float32).to(device)

    #initial condition 

    # Calculate position and velocity
    y = np.sin(t)
    v = np.cos(t)  # velocity = dy/dt

    # Add noise to make it more "real"
    y_noise = y + 0.1 * np.random.randn(n_samples)
    v_noise = v + 0.1 * np.random.randn(n_samples)

    # Stack into state: [position, velocity]
    state = np.stack([y_noise, v_noise], axis=1)  # Shape: [n_samples, 2]

    # Convert to tensors
    t_tensor = torch.tensor(t).float()
    state_tensor = torch.tensor(state).float()  # Shape: [n_samples, 2]

    return t_tensor, state_tensor

#True sprial dynamics
class SpiralDynamics(nn.Module):
    def __init__(self, device="cpu"):
        super(SpiralDynamics, self).__init__()

        #weight matrix for spiral
        self.A = torch.tensor([[-0.1, -1.0], 
                               [1.0, -0.1]], dtype=torch.float32, device=device)
        
    def forward(self, t, y):
        return torch.mm(y, self.A.T)

def generate_spiral(true_func, batch_size=32, n_samples=100, t_max=10, device="cpu"):
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
        true_trajectories = odeint(true_func, y0, t, method='dopri5')
        #Shape: [n_points, batch_size, 2]
    
    return t, y0, true_trajectories


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




# Testing generation
if __name__ == '__main__':
    #t, state = generate_irregular(50)
    #plot_samples(t, state, title="Test", file_name="test.png")

    # true_func = SpiralDynamics()
    # t, y0, true_traj = generate_spiral(true_func, batch_size=8, n_samples=100)
    # plot_spiral(t, y0, true_traj, title="True Spiral Dynamics", file_name="true_spirals.png")

    true_func = SineDynamics()
    t, y0, true_traj = generate_sine(true_func, batch_size=8, n_samples=100)
    plot_samples(t, y0, true_traj, title="True Sine Dynamics", file_name="true_sine.png")