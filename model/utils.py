import torch

def nan_detection(tensor):
    if torch.sum(torch.isinf(tensor)) != 0:
        return True
    if torch.sum(torch.isnan(tensor)) != 0:
        return True
    return False


# smoothing every list dim by
def smoothing(tensor, alpha=0.01):
    mean = torch.mean(tensor, dim=-1, keepdim=True)
    return (1 - alpha) * tensor + alpha * mean
