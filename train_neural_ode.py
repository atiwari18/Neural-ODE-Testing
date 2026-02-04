import torch
from models.neural_ode import ODEFunc, train_ode, plot_loss, extrapolate, plot_vector_field, plot_extrapolation
from dataset.sine import generate_irregular

#Load Data
print("Loading Data...")
t_train, y_train = generate_irregular(200)
t_train, _ = torch.sort(t_train)
print("Data Loaded!")

#Create model, optimizer and criterion
model = ODEFunc()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

#Training
print("\nTraining Neural ODE...")
losses = train_ode(model, 200, optimizer, criterion, t_train, y_train)

#Plotting loss
plot_loss(losses)
t_future, y_future = extrapolate(model, t_train, y_train)
plot_vector_field(model, file_name="Vector Field for Sine Wave (100 Samples).png")
plot_extrapolation(t_train, y_train, t_future, y_future, file_name="Extrapolation for Sine Wave (100 Samples).png")
