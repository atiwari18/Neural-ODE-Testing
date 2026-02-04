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


# Testing generation
if __name__ == '__main__':
    t, state = generate_irregular(50)
    plot_samples(t, state, title="Test", file_name="test.png")