import torch

def getDevice():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ' + str(device))
    return device