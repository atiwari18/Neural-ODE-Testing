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
    print("\nNeural ODE Predicted derivative:", pred_deriv)
    print("True derivative should be: [1.0, 0.0]")
    
    test_state2 = torch.tensor([[1.0, 0.0]]).to(device)  # position=1, velocity=0
    pred_deriv2 = node(torch.tensor(0.0), test_state2)
    print("Predicted derivative:", pred_deriv2)
    print("True derivative should be: [0.0, -1.0]")

#load Augmented Neural ODE
anode = AugmentedNODEFunc(time_invariant=True, augment_dim=1).to(device)
anode_weights = torch.load(".\\Results\\anode_sine_500-3.pth", weights_only=True)
anode.load_state_dict(anode_weights)

anode.eval()
with torch.no_grad():
    test_state = torch.tensor([[0.0, 1.0]]).to(device)
    test_state_aug = anode.augment(test_state)  # becomes [[0.0, 1.0, 0.0]]
    
    pred_deriv = anode(torch.tensor(0.0).to(device), test_state_aug)
    print("\nPredicted derivative:", pred_deriv)
    # Output will be [dy/dt, dv/dt, da/dt] — you only care about first two
    print("True derivative should be: [1.0, 0.0, ...]")

    test_state2 = torch.tensor([[1.0, 0.0]]).to(device)
    test_state2_aug = anode.augment(test_state2)  # becomes [[1.0, 0.0, 0.0]]
    
    pred_deriv2 = anode(torch.tensor(0.0).to(device), test_state2_aug)
    print("Predicted derivative:", pred_deriv2)
    print("True derivative should be: [0.0, -1.0, ...]")