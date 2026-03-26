import torch
print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
torch.cuda.empty_cache()