import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=2):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        #lstm layers
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )

        #output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Tanh(), 
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        output = self.fc(lstm_out)
        return output, hidden