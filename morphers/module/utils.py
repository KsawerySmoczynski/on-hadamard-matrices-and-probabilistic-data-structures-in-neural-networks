import torch


def get_optimizer(parameters: torch.ParameterDict):
    return torch.optim.AdamW(parameters, lr=3e-3, betas=[0.9, 0.999], weight_decay=0.01)
