import torch.nn as nn
import torch


def set_criterion(loss_type: str, loss_args):
    loss_class = {"YourFancyLoss": YourFancyLoss,
                  "L1Loss": nn.L1Loss,
                  "MSELoss": nn.MSELoss,
                  "BCE":    nn.BCELoss,
                  "CrossEntropyLoss": nn.CrossEntropyLoss}[loss_type]
    return loss_class(**loss_args)


class YourFancyLoss(nn.Module):
    def __init__(self, device):
        super(YourFancyLoss, self).__init__()
        self.device = device

    def forward(self, input, target):
        # Your loss algorithm here
        return torch.tensor(1.)
