import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import numpy as np
import os
 
class ODEFunc(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, time_invariant=False):
        super(ODEFunc, self).__init__()
        self.time_invariant = time_invariant

        #MLP for ODE
        if time_invariant:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),    
                nn.Tanh(), 
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim)     
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim + 1, hidden_dim),    
                nn.Tanh(), 
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim + 1)     
            )

        #weight and bias initialization from https://github.com/rtqichen/torchdiffeq
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, state):
        #time invariant behavior
        if not self.time_invariant:
            #batch_size
            batch = state.shape[0]

            #expand t to match batch_size [] --> [batch_size, 1]
            t_expanded = t.expand(batch, 1)

            state = torch.cat((state, t_expanded), dim=1)
        
        return self.net(state)

    
def train_ode(model, epochs, optimizer, criterion, true_traj, t, y0, file_name="neural_ode_sine.pth"):
    losses = []

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Integration from t_0 up to t_final
        #[n_samples, batch_size, 2]
        pred_state = odeint(model, y0, t, method='dopri5')

        loss = criterion(pred_state, true_traj)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    full_path = os.path.join(results_dir, file_name)

    torch.save(model.state_dict(), full_path)

    print(f"Model saved to {full_path}")

    return losses


def plot_loss(losses, file_name):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True)

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)    
    full_path = os.path.join(results_dir, file_name)

    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")


def extrapolate(model, t_train, state_train, device, t_max=6*torch.pi):
    t_future = torch.linspace(float(t_train[-1]), t_max, 100).to(device)

    with torch.no_grad():
        # Use last known state as the new initial condition
        state_future = odeint(model, state_train[-1:], t_future)

    return t_future, state_future

def plot_extrapolation(t_train, state_train, t_future, state_future, file_name, model, device):
    # Extract positions from states
    y_train = state_train[:, 0]
    y_future = state_future[:, :, 0]  # Shape: [time, batch, features]
    
    # Generate prediction from t=0 to show interpolation quality
    with torch.no_grad():
        t_interp = torch.linspace(0, float(t_train[-1]), 200).to(device)
        # Start from the initial state at t=0
        state_interp = odeint(model, state_train[0:1], t_interp)
        y_interp = state_interp[:, :, 0]  # Extract position
    
    # Generate Ground Truth for the whole range
    t_total = torch.linspace(0, 6 * np.pi, 200)
    y_total = torch.sin(t_total)
    
    plt.figure(figsize=(14, 6))
    
    # Plot Ground Truth
    plt.plot(t_total.numpy(), y_total.numpy(), color='gray', label='Ground Truth', linestyle='--', alpha=0.5, linewidth=2)
    
    # Plot Training Data (noisy observations)
    plt.scatter(t_train.cpu().numpy(), y_train.cpu().numpy(), color='red', label='Training Samples (Noisy)', s=30, zorder=5)
    
    # Plot Neural ODE interpolation (0 to end of training)
    plt.plot(t_interp.cpu().numpy(), y_interp.detach().cpu().numpy().squeeze(), 
             color='green', label='Neural ODE Fit (Training Region)', linewidth=2, alpha=0.8)
    
    # Plot Neural ODE extrapolation (beyond training)
    plt.plot(t_future.cpu().numpy(), y_future.detach().cpu().numpy().squeeze(), 
             color='blue', label='Neural ODE Extrapolation', linewidth=2)
    
    # Mark the boundary
    plt.axvline(x=t_train[-1].cpu().item(), color='black', linestyle=':', 
                linewidth=2, label='End of Training Data')
    
    plt.title("Neural ODE: Interpolation & Extrapolation (State-Space)")
    plt.xlabel("Time (t)")
    plt.ylabel("Position (y)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 6 * np.pi)
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    full_path = os.path.join(results_dir, file_name)
    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")


def plot_learned_dynamics_vs_true(model, device, file_name, y_range=(-1.5, 1.5), n_points=30):
    """
    For autonomous systems, we plot learned dynamics in phase space
    Shows: dv/dt vs y (should be -y) and dy/dt vs v (should be v)
    """
    model.eval()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Test range for position
    y_test = np.linspace(y_range[0], y_range[1], n_points)
    
    # Plot 1: dv/dt vs y (should be -y)
    with torch.no_grad():
        v_fixed = 0.0  # Fix velocity at 0
        states = torch.tensor([[y, v_fixed] for y in y_test]).float().to(device)
        derivatives = model(None, states).cpu().numpy()
        dv_dt_learned = derivatives[:, 1]  # Second component is dv/dt
    
    true_dv_dt = -y_test  # True: dv/dt = -y
    
    ax1.plot(y_test, dv_dt_learned, 'b-', linewidth=2, label='Learned dv/dt')
    ax1.plot(y_test, true_dv_dt, 'r--', linewidth=2, label='True dv/dt = -y')
    ax1.set_xlabel('Position (y)')
    ax1.set_ylabel('dv/dt')
    ax1.set_title('Acceleration vs Position (v=0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: dy/dt vs v (should be v)
    with torch.no_grad():
        y_fixed = 0.0  # Fix position at 0
        v_test = np.linspace(y_range[0], y_range[1], n_points)
        states = torch.tensor([[y_fixed, v] for v in v_test]).float().to(device)
        derivatives = model(None, states).cpu().numpy()
        dy_dt_learned = derivatives[:, 0]  # First component is dy/dt
    
    true_dy_dt = v_test  # True: dy/dt = v
    
    ax2.plot(v_test, dy_dt_learned, 'b-', linewidth=2, label='Learned dy/dt')
    ax2.plot(v_test, true_dy_dt, 'r--', linewidth=2, label='True dy/dt = v')
    ax2.set_xlabel('Velocity (v)')
    ax2.set_ylabel('dy/dt')
    ax2.set_title('Velocity Derivative (y=0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    full_path = os.path.join(results_dir, file_name)

    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")

#Plot spiral extrapolation
def plot_spiral_extrapolation(t_train, state_train, state_future, true_func=None, file_name=None, model=None, device=None):
    # Extract coordinates
    x_train = state_train[:, 0].cpu().numpy()
    y_train = state_train[:, 1].cpu().numpy()
    y0_train = state_train[0:1, :].to(device)
    
    x_future = state_future[:, 0, 0].cpu().numpy()
    y_future = state_future[:, 0, 1].cpu().numpy()
    
    plt.figure(figsize=(10, 10))

    # Generate TRUE ground truth using the ODE solver
    if true_func is not None and device is not None:
        with torch.no_grad():
            # Create dense time points for smooth ground truth
            t_train_min = t_train[0].item()
            t_train_max = t_train[-1].item()
            t_gt_dense = torch.linspace(t_train_min, t_train_max, 300).to(device)
            
            # Solve TRUE dynamics
            state_gt = odeint(true_func, y0_train, t_gt_dense)
            
            # Extract coordinates
            x_gt = state_gt[:, 0, 0].cpu().numpy()
            y_gt = state_gt[:, 0, 1].cpu().numpy()
            
            # Plot ground truth
            plt.plot(x_gt, y_gt, 'gray', linestyle='--', alpha=0.5, 
                    linewidth=2.5, label='True Dynamics (Ground Truth)')

    #if model is provided, compute the learned trajectory through the trainin region.
    if model is not None and device is not None:
        with torch.no_grad():
            #Create dense time points for a smooth trajectory
            t_train_min = t_train[0].item()
            t_train_max = t_train[-1].item()
            t_train_dense = torch.linspace(t_train_min, t_train_max, 200).to(device)
            
            # Integrate model through training region
            state_train_pred = odeint(model, state_train[0:1], t_train_dense)
            
            # Extract coordinates
            x_train_pred = state_train_pred[:, 0, 0].cpu().numpy()
            y_train_pred = state_train_pred[:, 0, 1].cpu().numpy()
            
            # Plot learned trajectory in training region
            plt.plot(x_train_pred, y_train_pred, 'green', linewidth=3, alpha=0.9, 
                    label='Learned Trajectory (Training Region)')
    else:
        # Fallback: just connect training points
        plt.plot(x_train, y_train, 'green', linewidth=3, alpha=0.9, label='Training Region (Connected Points)')
    
    # Training data points
    plt.scatter(x_train, y_train, c='red', s=40, alpha=0.7, zorder=5, label='Training Data')
    
    # Extrapolation
    plt.plot(x_future, y_future, 'blue', linewidth=3, label='Extrapolation')
    
    # Markers
    plt.scatter([x_train[0]], [y_train[0]], c='green', s=110, marker='o', 
               edgecolors='black', linewidth=2, label='Start', zorder=10)
    plt.scatter([x_train[-1]], [y_train[-1]], c='orange', s=110, marker='s', 
               edgecolors='black', linewidth=2, label='End of Training', zorder=10)
    
    plt.title("Neural ODE: Spiral Extrapolation", fontsize=14, fontweight='bold')
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, file_name), dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {os.path.join(results_dir, file_name)}")


def evaluate_on_holdout(model, criterion, t_test, data_test):
    """
    Evaluate the trained model on a hold-out test set.
    Uses the initial condition from test data and integrates through test timepoints.
    """
    model.eval()
    with torch.no_grad():
        # Integrate from first test point through all test timepoints
        pred_state = odeint(model, data_test[0:1], t_test)
        loss = criterion(pred_state, data_test)
    return loss.item()

def plot_comparison(true_func, learned_func, device, file_name="True vs. Learned Dynamics.png", n_trajectories=5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate test initial conditions
    theta = torch.linspace(0, 2*np.pi, n_trajectories+1)[:-1]
    y0 = torch.stack([
        2.0 * torch.cos(theta),
        2.0 * torch.sin(theta)
    ], dim=1).to(device)
    
    t = torch.linspace(0, 25, 200).to(device)
    
    # True dynamics
    with torch.no_grad():
        true_traj = odeint(true_func, y0, t, method='dopri5')
        pred_traj = odeint(learned_func, y0, t, method='dopri5')
    
    # Plot true spirals
    for i in range(n_trajectories):
        traj = true_traj[:, i, :].cpu().numpy()
        axes[0].plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.6)
        axes[0].scatter(traj[0, 0], traj[0, 1], c='green', s=50, zorder=5)
    
    axes[0].set_title("True Dynamics")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Plot learned spirals
    for i in range(n_trajectories):
        traj = pred_traj[:, i, :].cpu().numpy()
        axes[1].plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.6)
        axes[1].scatter(traj[0, 0], traj[0, 1], c='green', s=50, zorder=5)
    
    # Save
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, file_name), dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {os.path.join(results_dir, file_name)}")

def plot_sine_extrapolation(t_train, state_train, t_future, state_future, true_func=None, file_name=None, model=None, device=None):
    # Extract position (first dimension of state)
    t_train_np = t_train.cpu().numpy()
    y_train = state_train[:, 0].cpu().numpy()  # Position only
    
    t_future_np = t_future.cpu().numpy()
    #y_future = state_future[:, 0, 0].cpu().numpy()  # Position only
    
    y0_train = state_train[0:1, :].to(device)  # Initial condition [1, 2]
    
    plt.figure(figsize=(14, 6))
    
    # Generate TRUE ground truth using the ODE solver
    if true_func is not None and device is not None:
        with torch.no_grad():
            # Create dense time points for smooth ground truth
            t_train_min = t_train[0].item()
            t_train_max = t_train[-1].item()
            t_gt_dense = torch.linspace(t_train_min, t_train_max, 300).to(device)
            
            # Solve TRUE dynamics
            state_gt = odeint(true_func, y0_train, t_gt_dense)
            
            # Extract position (first dimension)
            t_gt_np = t_gt_dense.cpu().numpy()
            y_gt = state_gt[:, 0, 0].cpu().numpy()
            
            # Plot ground truth
            plt.plot(t_gt_np, y_gt, 'gray', linestyle='--', alpha=0.5, linewidth=2.5, label='True Dynamics (Ground Truth)')
    
    # Plot learned trajectory through training region
    if model is not None and device is not None:
        with torch.no_grad():
            #Generate single time span that covers BOTH training and future
            t_full = torch.linspace(t_train_np[0].item(), t_future_np[-1].item(), 500).to(device)
            
            # Integrate model through training region
            state_full = odeint(model, y0_train, t_full)
            
            # Extract position
            t_full_np = t_full.cpu().numpy()
            y_train_pred = state_full[:, 0, 0].cpu().numpy()
            
            # Plot learned trajectory in training region
            plt.plot(t_full_np, y_train_pred, 'green', linewidth=2.5, alpha=0.8, label='Learned Dynamics (Training Region)')
    
    # Training data points
    plt.scatter(t_train_np, y_train, c='red', s=40, alpha=0.7, zorder=5, label='Training Observations')
    
    # Mark boundaries
    plt.axvline(x=t_train_np[-1], color='orange', linestyle=':', linewidth=2, alpha=0.7, label='End of Training Data')
    
    plt.title("Neural ODE: Sine Wave Extrapolation", fontsize=14, fontweight='bold')
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Position (y)", fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, file_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {os.path.join(results_dir, file_name)}")