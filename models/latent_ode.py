import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
    
#Decoder (latent --> observation)
class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, hidden_dim=20):
        super(Decoder, self).__init__()

        #MLP to decode from latent space to data space
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 =nn.Linear(hidden_dim, obs_dim)
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

