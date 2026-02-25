import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint as odeint
import os
from models.neural_ode import ODEFunc

#Encoder
#This is a backwards RNN - it process the observations in reverse order to product an initial
#latent state that has information about the trajectory.
class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, hidden_dim=25):
        super(RecognitionRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        #Input to Hidden: Combines observation + previous hidden state
        self.i2h = nn.Linear(obs_dim + hidden_dim, hidden_dim)

        #Hidden to Output --> produces latent mean and log-variance
        self.h2o = nn.Linear(hidden_dim, latent_dim*2)

    def forward(self, x, h):
        #Concatenate observation with the hidden state 
        combine = torch.cat((x, h), dim=1)

        #Update the hidden state with the tanh activation
        h = torch.tanh(self.i2h(combine))
        
        #Produce the latent parameters
        out = self.h2o(h)

        return out, h
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim).to(device)
    
#Decoder (latent --> observation)
class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, hidden_dim=20):
        super(Decoder, self).__init__()

        #MLP to decode from latent space to data space
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, obs_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, z):
        #handle both 3d and 2d inputs
        original_shape = z.shape
        if len(z.shape) == 3:
            #reshape [time, batch, latent] --> [time*batch, latent]
            z = z.reshape(-1, z.shape[-1])

        #decode!!!
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)

        #Reshape back if needed
        if len(original_shape) == 3:
            out = out.reshape(original_shape[0], original_shape[1], -1)

        return out

class LatentODE(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, encoder_hidden=25, ode_hidden=20, decoder_hidden=20):
        super(LatentODE, self).__init__()

        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        #3 network components
        self.encoder = RecognitionRNN(latent_dim, obs_dim, encoder_hidden)
        self.ode_func = ODEFunc(latent_dim, ode_hidden, time_invariant=True)
        self.decoder = Decoder(latent_dim, obs_dim, decoder_hidden)

    def encode(self, observed_data, observed_times):
        """
        Enocder observation sequence into initial latent state z0.

        Processes the data BACKWARDS (from last to dirst) to get z0
        """
        batch_size = observed_data.shape[0]
        seq_len = observed_data.shape[1]

        #initialize hidden state
        h = self.encoder.init_hidden(batch_size, observed_data.device)

        #Process observation backwards through time, this allows the RNN to
        #accumulate information from future to past
        for t in reversed(range(seq_len)):
            obs = observed_data[:, t, :]
            out, h = self.encoder(obs, h)

        #split output into mean and log-variance
        z0_mean = out[:, :self.latent_dim]
        z0_logvar = out[:, self.latent_dim:]

        return z0_mean, z0_logvar
    
    def reparametrize(self, mean, logvar):
        #compute the standard deviation from logvar
        std = torch.exp(0.5 * logvar)

        #sample epsilon from standard normal
        eps = torch.randn_like(std)

        #Reparametrize = μ + σ * ε
        return mean + eps * std
    
    def forward(self, observed_data, observed_times, prediction_times):
        #Encode observations to get initial latent state
        z0_mean, z0_var = self.encode(observed_data, observed_times)

        #Sample latent using reparametrization trick
        z0 = self.reparametrize(z0_mean, z0_var)

        #Solve ODE in latent space
        z_traj = odeint(self.ode_func, z0, prediction_times, method="dopri5")

        #Decode latent trajectory to observations
        predicted_obs = self.decoder(z_traj)

        return predicted_obs, z0_mean, z0_var
    
def latent_ode_loss(predicted, target, z0_mean, z0_log_var, kl_weight=1.0):
    #Reconstruction loss
    recon_loss = torch.mean((predicted-target) ** 2)

    #KL Divergence = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + z0_log_var - z0_mean.pow(2) - z0_log_var.exp())

    #total loss 
    total_loss = recon_loss + kl_weight * kl_loss

    return total_loss, recon_loss, kl_loss

def train_latent_ode(model, trajs, t, epochs=200, lr=0.001, kl_weight=1, device="cuda", file_name="latent_ode.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #rearrange trajectories to be [batch, n, obs_dim]
    traj_batch = trajs.permute(1, 0, 2)
    
    losses = {"total": [], "recon": [], "kl": []}

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        #forward pass
        predicted, z0_mean, z0_logvar = model(traj_batch, t, t)

        #compute loss
        total_loss, recon_loss, kl_loss = latent_ode_loss(predicted, trajs, z0_mean, z0_logvar, kl_weight)

        #backward pass
        total_loss.backward()
        optimizer.step()

        #Record losses 
        losses["total"].append(total_loss.item())
        losses["recon"].append(recon_loss.item())
        losses["kl"].append(kl_loss.item())

        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Total: {total_loss.item():.6f} | "
                  f"Recon: {recon_loss.item():.6f} | "
                  f"KL: {kl_loss.item():.6f}")
            
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    
    results_dir = os.path.join(project_root, 'Results')
    os.makedirs(results_dir, exist_ok=True)
    full_path = os.path.join(results_dir, file_name)

    torch.save(model.state_dict(), full_path)

    print(f"Model saved to {full_path}")
            
    return losses

def plot_loss(losses, file_name="Latent ODE Losses.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses["total"], label='Training Loss')
    plt.title('Latent ODE Training Loss Over Epochs')
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

if __name__ == "__main__":
    from dataset.data import SineDynamics, generate_sine

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Generate data
    true_func = SineDynamics(device=device).to(device)
    t, y0, trajs = generate_sine(true_func, batch_size=16, n_samples=100, t_max=4*torch.pi, device=device)
    t = t.to(device)
    y0 = y0.to(device)
    trajs = trajs.to(device)

    #Create Model
    model = LatentODE(latent_dim=4,
                      obs_dim=2, 
                      encoder_hidden=25, 
                      ode_hidden=20,
                      decoder_hidden=20).to(device)
    
    #Train
    print("\nTraining Latent ODE...")
    losses = train_latent_ode(model, trajs, t, epochs=20, kl_weight=1, device=device)





