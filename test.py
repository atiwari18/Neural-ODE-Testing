import torch
from models.neural_ode import ODEFunc, AugmentedNODEFunc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#load Neural ODE
node = ODEFunc(time_invariant=True).to(device)
weights = torch.load(".\\Results\\neural_ode_sine_1000.pth", weights_only=True)
node.load_state_dict(weights)

node.eval()
with torch.no_grad():
    test_state = torch.tensor([[0.0, 1.0]]).to(device)  # position=0, velocity=1
    pred_deriv = node(torch.tensor(0.0), test_state)
    print("Predicted derivative:", pred_deriv)
    print("True derivative should be: [1.0, 0.0]")
    
    test_state2 = torch.tensor([[1.0, 0.0]]).to(device)  # position=1, velocity=0
    pred_deriv2 = node(torch.tensor(0.0), test_state2)
    print("Predicted derivative:", pred_deriv2)
    print("True derivative should be: [0.0, -1.0]")