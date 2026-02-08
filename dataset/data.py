import numpy as np 
import matplotlib.pyplot as plt
import torch
import os

def generate_irregular(n_samples, t_max=4*np.pi):
    t = np.sort(np.random.rand(n_samples) * t_max)
    t[0] = 0

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

def generate_spiral(n_samples, t_max=100):
    #timepoint
    t = np.linspace(0, t_max, n_samples)

    #Generate spiral
    x = np.sin(t) * np.exp(-0.1 * t)
    y = np.cos(t) * np.exp(-0.1 * t)

    #add noise
    x += np.random.normal(0, 0.01, size=t.shape)
    y += np.random.normal(0, 0.01, size=t.shape)

    data = np.stack([x, y], axis=1)
    data = torch.tensor(data).float()

    return data


def plot_samples(t, state, title, file_name):
    # Extract position from state
    y = state[:, 0]  # First column is position
    
    # Generate a smooth line for sine wave
    t_smooth = np.linspace(0, float(t.max()), 200)
    y_true = np.sin(t_smooth)

    plt.figure(figsize=(10, 5))

    # Plot ground truth
    plt.plot(t_smooth, y_true, label="True Function", color="red", linestyle="--", alpha=0.6)

    # Plot the irregular, noisy samples (only position)
    plt.scatter(t.numpy(), y.numpy(), color="blue", label="Irregular Observations", s=18)
    plt.title(title)
    plt.xlabel("Time (t)")
    plt.ylabel("Position (y)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    full_path = os.path.join(results_dir, file_name)

    # Save the figure
    plt.savefig(full_path)
    print(f"Plot saved to: {full_path}")

def plot(data, t_max, file_name="Sprial_Data.png", fig_size=(8, 8)):
    # Generate a smooth line for sine wave
    t_smooth = np.linspace(0, t_max, 200)
    x_true = np.sin(t_smooth) * np.exp(-0.1 * t_smooth)
    y_true = np.cos(t_smooth) * np.exp(-0.1 * t_smooth)
    data_true = np.stack([x_true, y_true], axis=1)


    plt.figure(figsize=fig_size)
    plt.plot(data_true[:, 0], data_true[:, 1], label="True Function", color="red", linestyle="--", alpha=0.6)
    plt.scatter(data[:, 0], data[:, 1], label="Spiral Trajectory", color="blue", s=16)
    plt.title("Generated Spiral Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid()

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

    data = generate_spiral(200, t_max=6.29*5)
    plot(data, t_max=6.29*5)