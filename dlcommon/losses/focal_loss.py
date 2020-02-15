import torch
import torch.nn as nn

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.bce_wll = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        if len(target.shape)==1:
            logp = self.ce(input, target)
        else:
            logp = self.bce_wll(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()