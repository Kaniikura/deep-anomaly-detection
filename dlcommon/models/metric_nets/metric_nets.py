from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from .metrics import AdaCosProduct, ArcFaceProduct, CosFaceProduct, SphereFaceProduct
from .encoder import build_encoder


class FeatureExtractor(nn.Module):
    def __init__(self, image_size, encoder, num_features):
        super().__init__()

        self.encoder = build_encoder(encoder)
        self._out_channel = self.encoder.out_channel
        _out_h = _out_w = math.ceil(image_size // 32)
        self.out_features = np.prod(np.array([_out_h, _out_w, self._out_channel]))

    def forward(self, input):
            x = self.encoder(input)
            x = x.view(-1, self.out_features)

            return x


class xxxFace(nn.Module):
    def __init__(self, image_size, encoder, num_features, num_classes, xface_product):
        super(xxxFace, self).__init__()
        self.encoder = FeatureExtractor(image_size, encoder, num_features)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.encoder.out_features, num_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_features),
        )
        self.product = xface_product(num_features, num_classes)
    
    def get_feature(self, input):
        x = self.encoder(input)
        x = self.fc(x)

        return x
        
    def forward(self, input, label=None):
        x = self.get_feature(input)
        x = self.product(x, label)
        
        return x


class ArcFace(xxxFace):
    def __init__(self, image_size, encoder, num_features, num_classes, xface_product=ArcFaceProduct):
        super(ArcFace, self).__init__(image_size, encoder, num_features, num_classes, xface_product)

class AdaCos(xxxFace):
    def __init__(self, image_size, encoder, num_features, num_classes, xface_product=AdaCosProduct):
        super(AdaCos, self).__init__(image_size, encoder, num_features, num_classes, xface_product=AdaCosProduct)

class CosFace(xxxFace):
    def __init__(self, image_size, encoder, num_features, num_classes, xface_product=CosFaceProduct):
        super(CosFace, self).__init__(image_size, encoder, num_features, num_classes, xface_product)

class SphereFace(xxxFace):
    def __init__(self, image_size, encoder, num_features, num_classes, xface_product=SphereFaceProduct):
        super(SphereFace, self).__init__(image_size, encoder, num_features, num_classes, xface_product)


class VanillaCNN(nn.Module):
    def __init__(self, image_size, encoder, num_features, num_classes):
        super(VanillaCNN, self).__init__()
        self.encoder = FeatureExtractor(image_size, encoder, num_features)
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.encoder.out_features, num_features),
            nn.ReLU(),
            nn.BatchNorm1d(num_features),
        )
        self.product = nn.Sequential(
            nn.Linear(num_features, num_classes),
        )
    
    def get_feature(self, input):
        x = self.encoder(input)
        x = self.fc(x)

        return x
        
    def forward(self, input, label=None):
        x = self.get_feature(input)
        x = self.product(x)
        
        return x
