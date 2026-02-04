import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import numpy as np
import os

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init()

        #MLP
        self.net = nn.Sequential(
            nn.Linear(2, 25),        #2 here because we are concatenating time. 
            nn.Tanh(), 
            nn.Linear(25, 1)
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
        pred_y = odeint(model, y[0], t)

        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        print(f"Epoch #{epoch} | Loss: {loss.item()}")

    return losses

def extrapolate(model, t_train, y_train):
    t_future = torch.linspace(float(t_train[-1]), 6 * torch.pi, 50)

    with torch.no_grad():
        #Use last known point as the new initial condition
        y_future = odeint(model, y_train[-1], t_future)

    return t_future, y_future

def plot_vector_field(model, file_name, t_range=(0, 6*np.pi), y_range=(-1.5, 1.5)):
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
                t_val = torch.tensor([[t_grid[i]]]).float()
                y_val = torch.tensor([[y_grid[j]]]).float()

                # The model outputs the derivative
                dy_dt = model(t_val, y_val)
                V[j, i] = dy_dt.item()

    plt.figure(figsize=(10, 6))
    plt.streamplot(T, Y, U, V, color=V, cmap='coolwarm', alpha=0.8)
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

def plot_extrapolation(t_train, y_train, t_future, y_future, file_name):
    #Generate Ground Truth for the whole range
    t_total = torch.linspace(0, 6 * np.pi, 200)
    y_total = torch.sin(t_total)

    plt.figure(figsize=(12, 6))
    
    #Plot Ground Truth
    plt.plot(t_total.numpy(), y_total.numpy(), color='gray', label='Ground Truth', linestyle='--', alpha=0.5)
    
    #Plot Training Data
    plt.scatter(t_train.numpy(), y_train.numpy(), color='red', label='Training Samples (Noisy)', s=20)
    
    #Plot Extrapolation
    #Note: we squeeze because odeint output is [time, batch, dim]
    plt.plot(t_future.numpy(), y_future.detach().numpy().squeeze(), 
             color='blue', label='Neural ODE Extrapolation', linewidth=2)

    plt.axvline(x=t_train[-1].item(), color='black', linestyle=':', label='End of Training Data')
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