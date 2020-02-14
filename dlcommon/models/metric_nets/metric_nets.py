from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import Parameter
from .encoder import build_encoder


class FeatureExtractor(nn.Module):
    def __init__(self, encoder, num_features):
        super().__init__()

        self.encoder = build_encoder(encoder)
        self.enc_channels = self.encoder.channels
        print(self.encoder.channels)

        #self.fc = nn.Linear(self.enc_channels, num_features)
        #self.bn = nn.BatchNorm1d(num_features)

    def forward(self, input):
            x = self.encoder(input)
            #x = self.fc(x)
            #x = self.bn(x)
            
            output = x
            
            return output


class ArcFace(nn.Module):
    def __init__(self, encoder, num_features, num_classes, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.encoder = FeatureExtractor(encoder, num_features)

        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input):
    #def get_feature(self, input):
        output = self.encoder(input)

        return output

    """def forward(self, input, label=None):
        x = get_feature(input)
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output"""