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
    
def train_lstm(lstm, epochs, optimizer, criterion, inputs, targets, device):
    n_windows = inputs.shape[0]
    losses = []

    #training loop
    lstm.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        #shuffle order of windows each epoch
        perm = torch.randperm(n_windows)

        for i in perm:
            #get batch
            x = inputs[i].to(device)
            y = targets[i].to(device)

            optimizer.zero_grad()

            #forward pass
            pred, _ = lstm(x)

            #loss
            loss = criterion(pred, y)

            #backwards
            loss.backward()
            optimizer.step()

            epoch.loss += loss.item()

        #calculate the average loss
        avg_loss = epoch_loss / n_windows
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

        return losses