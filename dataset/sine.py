import numpy as np 
import matplotlib.pyplot as plt
import torch
import os

def generate_irregular(n_samples, t_max=4*np.pi):
    t = np.sort(np.random.rand(n_samples) * 4 * np.pi)
    t[0] = 0

    #calculate the sine values
    y = np.sin(t)

    #Add noise to make it more "real"
    y_noise = y + 0.1 * np.random.randn(n_samples)

    #Convert to tensors
    t_tensor = torch.tensor(t).float()
    y_tensor = torch.tensor(y_noise).float().unsqueeze(-1) #Shape: [n_samples, feature_dim (1)]

    return t_tensor, y_tensor

def plot_samples(t, y, title, file_name):
    #Generate a smooth line for sine wave
    t_smooth = np.linspace(0, float(t.max()), 200)
    y_true = np.sin(t_smooth)

    plt.figure(figsize=(10, 5))

    #plot ground truth
    plt.plot(t_smooth, y_true, label="True Function", color="red", linestyle = "--", alpha=0.6)

    #PLot the regular, noisy samples
    plt.scatter(t.numpy(), y.numpy(), color="blue", label="Irregular Observations", s=18)
    plt.title(title)
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


#Testing generation
if __name__ == '__main__':
    t, y = generate_irregular(50)
    plot_samples(t, y, title="Test", file_name="test.png")