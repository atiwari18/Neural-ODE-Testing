import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import numpy as np
import os
 
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ODEFunc, self).__init__()

        # MLP that takes state [y, v] and outputs derivatives [dy/dt, dv/dt]
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),    # Input: [y, v]
            nn.Tanh(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)     # Output: [dy/dt, dv/dt]
        )

    def forward(self, t, state):
        # state shape: [batch_size, 2] where state = [y, v]
        # Output: [dy/dt, dv/dt]
        # We ignore t because this is an autonomous system
        return self.net(state)

    
def train_ode(model, epochs, optimizer, criterion, t, state):
    losses = []

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Integration from t_0 up to t_final
        pred_state = odeint(model, state[0:1], t)

        loss = criterion(pred_state, state)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 20 == 0:  # Print less frequently
            print(f"Epoch #{epoch} | Loss: {loss.item():.6f}")

    return losses


def plot_loss(losses):
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
    
    full_path = os.path.join(results_dir, "Losses.png")

    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")


def extrapolate(model, t_train, state_train, device):
    t_future = torch.linspace(float(t_train[-1]), 6 * torch.pi, 50).to(device)

    with torch.no_grad():
        # Use last known state as the new initial condition
        state_future = odeint(model, state_train[-1:], t_future)

    return t_future, state_future


def plot_vector_field(model, file_name, device, t_range=(0, 6*np.pi), y_range=(-1.5, 1.5), v_range=(-1.5, 1.5)):
    """
    For autonomous systems, we plot the phase space (y vs v), not (t vs y)
    """
    y_grid = np.linspace(y_range[0], y_range[1], 20)
    v_grid = np.linspace(v_range[0], v_range[1], 20)
    Y, V = np.meshgrid(y_grid, v_grid)

    # Calculate derivatives at each point
    dY = np.zeros_like(Y)  # dy/dt
    dV = np.zeros_like(V)  # dv/dt

    model.eval()
    with torch.no_grad():
        for i in range(len(y_grid)):
            for j in range(len(v_grid)):
                state = torch.tensor([[y_grid[i], v_grid[j]]]).float().to(device)
                
                # The model outputs [dy/dt, dv/dt]
                derivatives = model(None, state)  # t is ignored
                dY[j, i] = derivatives[0, 0].cpu().item()
                dV[j, i] = derivatives[0, 1].cpu().item()

    plt.figure(figsize=(10, 8))
    plt.streamplot(Y, V, dY, dV, color=np.sqrt(dY**2 + dV**2), cmap='viridis')
    plt.title("Phase Space: Learned Vector Field")
    plt.xlabel("Position (y)")
    plt.ylabel("Velocity (v)")
    plt.colorbar(label='Magnitude of derivative')
    plt.grid(True, alpha=0.3)
    
    # Add a circle to show the true trajectory
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.sin(theta), np.cos(theta), 'r--', alpha=0.5, label='True trajectory (circle)')
    plt.legend()
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    full_path = os.path.join(results_dir, file_name)

    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")


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