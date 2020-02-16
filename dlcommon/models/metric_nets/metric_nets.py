from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from .metrics import AdaCosProduct, ArcFaceProduct, CosFaceProduct, SphereFaceProduct
from .encoder import build_encoder


class FeatureExtractor(nn.Module):
    def __init__(self, encoder, num_features):
        super().__init__()

        self.encoder = build_encoder(encoder)
        self._out_shape = self.encoder.out_shape
        self.out_features = np.prod(np.array(self._out_shape))

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        #self.fc = nn.Linear(self.out_features, num_features)

    def forward(self, input):
            x = self.encoder(input)
            #x = x.view(-1, self.out_features)
            x = self.fc(x)
            output = x

            return output


class xxxFace(nn.Module):
    def __init__(self, encoder, num_features, num_classes, xface_product):
        super(xxxFace, self).__init__()
        self.encoder = FeatureExtractor(encoder, num_features)
        #self.product = xface_product(num_features, num_classes)
        self.product = nn.Sequential(
            nn.Linear(512, 15)
        )
    
    def get_feature(self, input):
        x = self.encoder(input)

        return x
        
    def forward(self, input, label=None):
        x = self.get_feature(input)
        x = self.product(x)
        #x = self.product(x, label)
        
        return x


class ArcFace(xxxFace):
    def __init__(self, encoder, num_features, num_classes, xface_product=ArcFaceProduct):
        super(ArcFace, self).__init__(encoder, num_features, num_classes, xface_product)

class AdaCos(xxxFace):
    def __init__(self, encoder, num_features, num_classes, xface_product=AdaCosProduct):
        super(AdaCos, self).__init__(encoder, num_features, num_classes, xface_product)

class CosFace(xxxFace):
    def __init__(self, encoder, num_features, num_classes, xface_product=CosFaceProduct):
        super(CosFace, self).__init__(encoder, num_features, num_classes, xface_product)

class SphereFace(xxxFace):
    def __init__(self, encoder, num_features, num_classes, xface_product=SphereFaceProduct):
        super(SphereFace, self).__init__(encoder, num_features, num_classes, xface_product)