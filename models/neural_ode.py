import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import numpy as np
import os
 
class AugmentedNODEFunc(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, 
                 time_invariant=False, augment_dim=0):
        super(AugmentedNODEFunc, self).__init__()
        self.time_invariant = time_invariant
        self.augment_dim = augment_dim
        self.input_dim = input_dim
        self.nfe = 0

        # Network operates on augmented state
        effective_dim = input_dim + augment_dim

        if time_invariant:
            self.net = nn.Sequential(
                nn.Linear(effective_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, effective_dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(effective_dim + 1, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, effective_dim + 1)
            )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, state):
        # state is already augmented — odeint calls this directly
        self.nfe += 1

        if not self.time_invariant:
            batch = state.shape[0]
            t_expanded = t.expand(batch, 1)
            state = torch.cat((state, t_expanded), dim=1)

        return self.net(state)

    def augment(self, y0):
        """Pad initial conditions with zeros for augmented dims."""
        zeros = torch.zeros(y0.shape[0], self.augment_dim, device=y0.device)
        return torch.cat([y0, zeros], dim=1)

    def strip(self, state):
        """Remove augmented dims from output, returning original state dims only."""
        return state[:, :, :self.input_dim]

    def reset_nfe(self):
        self.nfe = 0

class ODEFunc(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, time_invariant=False):
        super(ODEFunc, self).__init__()
        self.time_invariant = time_invariant
        self.nfe = 0

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
        self.nfe += 1

        #time invariant behavior
        if not self.time_invariant:
            #batch_size
            batch = state.shape[0]

            #expand t to match batch_size [] --> [batch_size, 1]
            t_expanded = t.expand(batch, 1)

            state = torch.cat((state, t_expanded), dim=1)
        
        return self.net(state)
    
    def reset_nfe(self):
        self.nfe = 0
        return

    
def train_ode(model, epochs, optimizer, criterion, true_traj, t, y0, reg=0.01, file_name="neural_ode_sine.pth"):
    losses = []
    is_anode = isinstance(model, AugmentedNODEFunc)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        y0_in = model.augment(y0) if is_anode else y0

        # Integration from t_0 up to t_final
        #[n_samples, batch_size, 2]
        pred_state = odeint(model, y0_in, t, method='dopri5')

        if is_anode: 
            #extract augmented dimensions BEFORE stripping
            aug_dims = pred_state[:, :, model.input_dim:]         #[T, batch, augment_dim]
            aug_penalty = (aug_dims ** 2).mean()

            pred_state = model.strip(pred_state)
            loss = criterion(pred_state, true_traj) + (reg * aug_penalty)
        else:
            loss = criterion(pred_state, true_traj)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            if is_anode:
                print(f"Epoch {epoch+1}/{epochs} | "
                    f"Loss: {loss.item():.6f} | "
                    f"MSE: {criterion(pred_state, true_traj).item():.6f} | "
                    f"Aug penalty: {aug_penalty.item():.6f}")
            else:
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
    t_full = torch.linspace(float(t_train[0]), t_max, 500).to(device)
    is_anode = isinstance(model, AugmentedNODEFunc)

    model.reset_nfe() #reset before solving
    with torch.no_grad():
        y0_in = model.augment(state_train[0:1]) if is_anode else state_train[0:1]
        # Use last known state as the new initial condition
        state_full = odeint(model, y0_in, t_full)

        if is_anode:
            state_full = model.strip(state_full)

    nfe = model.nfe

    return t_full, state_full, nfe

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

def plot_comparison(true_func, learned_func, device, y0, file_name="True vs. Learned Dynamics.png", n_trajectories=5):
    fig, axes = plt.subplots(1, 2, figsize=(28, 12))
    
    # Use actual training initial conditions, not manufactured ones
    y0_subset = y0[:n_trajectories].to(device)
    
    # Compute radii for color coding and legend
    radii = torch.sqrt(y0_subset[:, 0]**2 + y0_subset[:, 1]**2).cpu().numpy()
    
    t = torch.linspace(0, 6 * np.pi, 500).to(device)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_trajectories))

    is_anode = isinstance(learned_func, AugmentedNODEFunc)
    
    with torch.no_grad():
        true_traj = odeint(true_func, y0_subset, t, method='dopri5')

        if is_anode:
            y0_aug = learned_func.augment(y0_subset)
            pred_traj_full = odeint(learned_func, y0_aug, t, method='dopri5')
            pred_traj = pred_traj_full[:, :, :learned_func.input_dim]
        else:
            pred_traj = odeint(learned_func, y0_subset, t, method='dopri5')
    
    for i in range(n_trajectories):
        true_np = true_traj[:, i, :].cpu().numpy()
        pred_np = pred_traj[:, i, :].cpu().numpy()
        
        axes[0].plot(true_np[:, 0], true_np[:, 1], '-', color=colors[i],
                    alpha=0.85, linewidth=1.8, label=f'r={radii[i]:.2f}')
        axes[0].scatter(true_np[0, 0], true_np[0, 1], color=colors[i], s=60,
                       zorder=5, edgecolors='black', linewidth=0.8)
        
        axes[1].plot(pred_np[:, 0], pred_np[:, 1], '-', color=colors[i],
                    alpha=0.85, linewidth=1.8, label=f'r={radii[i]:.2f}')
        axes[1].scatter(pred_np[0, 0], pred_np[0, 1], color=colors[i], s=60,
                       zorder=5, edgecolors='black', linewidth=0.8)
    
    for ax, title in zip(axes, ["True Dynamics (Phase Portrait)", 
                                 "Learned Dynamics (Phase Portrait)"]):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Position (y)", fontsize=11)
        ax.set_ylabel("Velocity (v)", fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.suptitle("Neural ODE Phase Space: Harmonic Oscillator\n"
                 "(Spiral drift = trajectory crossing failure)",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    plt.savefig(os.path.join(results_dir, file_name), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {os.path.join(results_dir, file_name)}")

def plot_sine_extrapolation(t_train, state_train, t_full, state_full, true_func=None, file_name=None, model=None, device=None):
    t_train_np = t_train.cpu().numpy()
    y_train = state_train[:, 0].cpu().numpy()
    y0_train = state_train[0:1, :].to(device)

    t_full_np = t_full.cpu().numpy()
    y_full    = state_full[:, 0, 0].cpu().numpy()

    # Split the single trajectory at the training boundary for coloring
    t_end       = t_train_np[-1]
    train_mask  = t_full_np <= t_end
    extrap_mask = t_full_np >= t_end

    plt.figure(figsize=(14, 6))

    # Ground truth over full range
    if true_func is not None and device is not None:
        with torch.no_grad():
            t_gt = torch.linspace(t_full[0].item(), t_full[-1].item(), 500).to(device)
            state_gt = odeint(true_func, y0_train, t_gt)
            plt.plot(t_gt.cpu().numpy(), state_gt[:, 0, 0].cpu().numpy(),
                     'gray', linestyle='--', alpha=0.5, linewidth=2.5,
                     label='True Dynamics (Ground Truth)')

    # Single continuous model trajectory — coloured green (train) then blue (extrap)
    plt.plot(t_full_np[train_mask],  y_full[train_mask],  'green', linewidth=2.5, alpha=0.8,
             label='Learned Dynamics (Training Region)')
    plt.plot(t_full_np[extrap_mask], y_full[extrap_mask], 'blue',  linewidth=2.5, alpha=0.8,
             label='Extrapolation')

     # Training observations
    plt.scatter(t_train_np, y_train, c='red', s=40, alpha=0.7,
                zorder=5, label='Training Observations')
    plt.axvline(x=t_end, color='orange', linestyle=':',
                linewidth=2, alpha=0.7, label='End of Training Data')

    plt.title("Neural ODE: Sine Wave Extrapolation", fontsize=14, fontweight='bold')
    plt.xlabel("Time (t)", fontsize=12)
    plt.ylabel("Position (y)", fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)

    script_path = os.path.abspath(__file__)
    results_dir = os.path.join(os.path.dirname(os.path.dirname(script_path)), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, file_name), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {os.path.join(results_dir, file_name)}")