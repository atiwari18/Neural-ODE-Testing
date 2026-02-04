import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import numpy as np
import os
 
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        #MLP
        self.net = nn.Sequential(
            nn.Linear(3, 50),        #3 here because we are concatenating periodic time function. 
            nn.Tanh(), 
            nn.Linear(50, 1)
        )

    def forward(self, t, y):
        #t is a scalar we need it to match ys shape.
        t_vec = torch.ones_like(y) * t

        #Concatenate
        concatenated = torch.cat([y, t_vec], dim=1)

        return self.net(concatenated)

    
def train_ode(model, epochs, optimizer, criterion, t, y):
    losses = []

    #Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        #Integration from t_0 up to t_final, the solcer will solve at every timestamp in t_train
        pred_y = odeint(model, y[0:1], t)

        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        print(f"Epoch #{epoch} | Loss: {loss.item()}")

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

    #Save the figure
    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")

def extrapolate(model, t_train, y_train, device):
    t_future = torch.linspace(float(t_train[-1]), 6 * torch.pi, 50).to(device)

    with torch.no_grad():
        #Use last known point as the new initial condition
        y_future = odeint(model, y_train[-1:], t_future)    #[-1:] so that the shape is (1, 1)

    return t_future, y_future

def plot_vector_field(model, file_name, device, t_range=(0, 6*np.pi), y_range=(-1.5, 1.5)):
    #create a grid of points
    t_grid = np.linspace(t_range[0], t_range[1], 20)
    y_grid = np.linspace(y_range[0], y_range[1], 20)
    T, Y = np.meshgrid(t_grid, y_grid)

    # Calculate gradients (dy/dt) for each point in the grid
    U = np.ones_like(T) # Time always moves forward at rate 1
    V = np.zeros_like(Y) # This will hold our dy/dt

    model.eval()
    with torch.no_grad():
        for i in range(len(t_grid)):
            for j in range(len(y_grid)):
                t_val = torch.tensor([[t_grid[i]]]).float().to(device)
                y_val = torch.tensor([[y_grid[j]]]).float().to(device)

                # The model outputs the derivative
                dy_dt = model(t_val, y_val)
                V[j, i] = dy_dt.item()

    plt.figure(figsize=(10, 6))
    plt.streamplot(T, Y, U, V, color=V, cmap='coolwarm')
    plt.title("Learned Vector Field (Time vs. State)")
    plt.xlabel("Time (t)")
    plt.ylabel("Value (y)")
    plt.colorbar(label='dy/dt (Slope)')
    plt.grid(True, alpha=0.3)
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    full_path = os.path.join(results_dir, file_name)

    #Save the figure
    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")

    return

def plot_extrapolation(t_train, y_train, t_future, y_future, file_name, device):
    #Generate Ground Truth for the whole range
    t_total = torch.linspace(0, 6 * np.pi, 200)
    y_total = torch.sin(t_total)

    plt.figure(figsize=(12, 6))
    
    #Plot Ground Truth
    plt.plot(t_total.numpy(), y_total.numpy(), color='gray', label='Ground Truth', linestyle='--', alpha=0.5)
    
    #Plot Training Data
    plt.scatter(t_train.cpu().numpy(), y_train.cpu().numpy(), color='red', label='Training Samples (Noisy)', s=20)
    
    #Plot Extrapolation
    #Note: we squeeze because odeint output is [time, batch, dim]
    plt.plot(t_future.cpu().numpy(), y_future.detach().cpu().numpy().squeeze(), 
             color='blue', label='Neural ODE Extrapolation', linewidth=2)

    plt.axvline(x=t_train[-1].cpu().item(), color='black', linestyle=':', label='End of Training Data')
    plt.title("Neural ODE: Beyond the Training Horizon")
    plt.xlabel("Time (t)")
    plt.ylabel("Value (y)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    full_path = os.path.join(results_dir, file_name)

    #Save the figure
    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")

    return

def plot_learned_dynamics_vs_true(model, device, t_train_end, t_max=30.0, n_points=500, y_dummy_value=0.0, figsize=(10, 6), file_name=None):
    model.eval()
    with torch.no_grad():
        t_test = torch.linspace(0, t_max, n_points).unsqueeze(1).to(device)   # shape: (N, 1)
        y_dummy = torch.full_like(t_test, y_dummy_value)
        inputs = torch.cat([y_dummy, t_test], dim=1)
        predicted_dydt = model.net(inputs).squeeze().cpu().numpy()
    
    true_dydt = np.cos(t_test.squeeze().cpu().numpy())
    
    plt.figure(figsize=figsize)
    plt.plot(t_test.cpu().numpy(), predicted_dydt, label='Learned dy/dt = f(y,t)', linewidth=2)
    plt.plot(t_test.cpu().numpy(), true_dydt, '--', label='True cos(t)', linewidth=1.5)
    plt.axvline(x=float(t_train_end), color='red', linestyle=':', 
                label='End of training data', alpha=0.7)
    
    plt.title("What the Neural ODE Actually Learned: dy/dt vs Time")
    plt.xlabel("Time t")
    plt.ylabel("dy/dt")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    full_path = os.path.join(results_dir, file_name)

    #Save the figure
    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")