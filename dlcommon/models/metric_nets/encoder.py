from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map, get_model_params, round_filters

from dlcommon.utils import build_from_config
from dlcommon.registry import BACKBONES


class ResNetEncoder(nn.Module):
    def __init__(self, backbone, **_):
        super().__init__()
        self.backbone = backbone
        if hasattr(self.backbone, 'avg_pool'):
            del self.backbone.avg_pool
        else:
            del self.backbone.avgpool
        del self.backbone.fc
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)

        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)

        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x

class PretrainedModelsResNextEncoder(nn.Module):
    def __init__(self, backbone, **_):
        super().__init__()
        self.backbone = backbone
        del self.backbone.avg_pool
        del self.backbone.last_linear
        
    def forward(self, x):
        x = self.backbone.features(x)

        return x



class DenseNetEncoder(nn.Module):
    def __init__(self, backbone, **_):
        super().__init__()
        self.backbone = backbone
        del self.backbone.classifier

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone.features.conv0(x)
        x = self.backbone.features.norm0(x)
        x = self.backbone.features.relu0(x)

        x = self.backbone.features.pool0(x)
        x = self.backbone.features.denseblock1(x)
        x = self.backbone.features.transition1(x)

        x = self.backbone.features.denseblock2(x)
        x = self.backbone.features.transition2(x)

        x = self.backbone.features.denseblock3(x)
        x = self.backbone.features.transition3(x)

        x = self.backbone.features.denseblock4(x)
        x = self.backbone.features.norm5(x)

        return x


class SENetEncoder(nn.Module):
    def __init__(self, backbone, **_):
        super().__init__()

        new_layer0 = [
                ('conv1', backbone.layer0.conv1),
                ('bn1', backbone.layer0.bn1),
                ('relu1', backbone.layer0.relu1)
        ]

        backbone.layer0 = nn.Sequential(OrderedDict(new_layer0))
        self.backbone = backbone
       

    def forward(self, x):
        x = self.backbone.layer0(x)
        x = self.backbone.layer1(F.max_pool2d(x, 2))
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        return x


class EfficientNetEncoder(EfficientNet):
    def __init__(self, skip_connections, model_name):
        blocks_args, global_params = get_model_params(model_name, override_params=None)

        super().__init__(blocks_args, global_params)
        self._skip_connections = list(skip_connections)
        self._skip_connections.append(len(self._blocks))
        
        del self._fc
        
    def forward(self, x):
        x = F.relu(self._bn0(self._conv_stem(x)))

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        
        return x

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('_fc.bias')
        state_dict.pop('_fc.weight')
        super().load_state_dict(state_dict, **kwargs)


def build_encoder(backbone):
    backbone_name = backbone

    if 'wsl' in backbone_name:
        backbone_config = {'name': backbone_name[:-4]}
    else:
        backbone_config = {'name': backbone_name, 'params': {'pretrained': 'imagenet'}}

    backbone = build_from_config(backbone_config, BACKBONES)

    if backbone_name.startswith('se'):
        return SENetEncoder(backbone)
    elif 'wsl' in backbone_name:
        if any(x in backbone_name for x in ['101_32x4d','101_64x4d']): 
            encoder = PretrainedModelsResNextEncoder(backbone) # pytorch-pretrainedmodels
        else: 
            encoder = ResNetEncoder(backbone) # torchvision models
        encoder.out_channel = BACKBONE_OUT_SHAPE[backbone_name]
    elif backbone_name.startswith('resnet'):
        encoder = ResNetEncoder(backbone)
    elif backbone_name.startswith('densenet'):
        encoder = DenseNetEncoder(backbone)
    elif backbone_name.startswith('eff'):
        encoder = EfficientNetEncoder(**efficient_net_encoders[backbone_name]['params'])
        settings = efficient_net_encoders[backbone_name]['pretrained_settings']['imagenet']
        encoder.load_state_dict(model_zoo.load_url(settings['url']))
        encoder.out_channel = efficient_net_encoders[backbone_name]['out_channel']

    if backbone_name in BACKBONE_OUT_CHANNEL:
        encoder.out_channel = BACKBONE_OUT_CHANNEL[backbone_name]

    return encoder

# Input assumes image　shape (3, 256, 256),
    # If you use a different size , modify the last two values as ceil(image_size/32).
BACKBONE_OUT_CHANNEL = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'resnext50_32x4d_wsl': 2048,
    'resnext101_32x4d_wsl': 2048,            
    'resnext101_64x4d_wsl': 2048,  
    ## WIP
    #'resnext101_32x8d_wsl': 2048, 
    #'resnext101_32x16d_wsl': 2048,
    #'resnext101_32x32d_wsl': 2048,
    #'resnext101_32x48d_wsl': 2048,
    'densenet121': 1024,
    'densenet169': 1664,
    'densenet201': 1920,
    'densenet161': 2208,
}


def _get_pretrained_settings(encoder):
    pretrained_settings = {
        'imagenet': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'url': url_map[encoder],
            'input_space': 'RGB',
            'input_range': [0, 1]
        }
    }
    return pretrained_settings


efficient_net_encoders = {
    'efficientnet-b0': {
        'encoder': EfficientNetEncoder,
        'out_channel': (320, 112, 40, 24, 32),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b0'),
        'params': {
            'skip_connections': [3, 5, 9],
            'model_name': 'efficientnet-b0'
        }
    },
    'efficientnet-b1': {
        'encoder': EfficientNetEncoder,
        'out_channel': (320, 112, 40, 24, 32),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b1'),
        'params': {
            'skip_connections': [5, 8, 16],
            'model_name': 'efficientnet-b1'
        }
    },
    'efficientnet-b2': {
        'encoder': EfficientNetEncoder,
        'out_channel': (352, 120, 48, 24, 32),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b2'),
        'params': {
            'skip_connections': [5, 8, 16],
            'model_name': 'efficientnet-b2'
        }
    },
    'efficientnet-b3': {
        'encoder': EfficientNetEncoder,
        'out_channel': (384, 136, 48, 32, 40),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b3'),
        'params': {
            'skip_connections': [5, 8, 18],
            'model_name': 'efficientnet-b3'
        }
    },
    'efficientnet-b4': {
        'encoder': EfficientNetEncoder,
        'out_channel': (448, 160, 56, 32, 48),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b4'),
        'params': {
            'skip_connections': [6, 10, 22],
            'model_name': 'efficientnet-b4'
        }
    },
    'efficientnet-b5': {
        'encoder': EfficientNetEncoder,
        'out_channel': (512, 176, 64, 40, 48),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b5'),
        'params': {
            'skip_connections': [8, 13, 27],
            'model_name': 'efficientnet-b5'
        }
    },
    'efficientnet-b6': {
        'encoder': EfficientNetEncoder,
        'out_channel': (576, 200, 72, 40, 56),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b6'),
        'params': {
            'skip_connections': [9, 15, 31],
            'model_name': 'efficientnet-b6'
        }
    },
    'efficientnet-b7': {
        'encoder': EfficientNetEncoder,
        'out_channel': (640, 224, 80, 48, 64),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b7'),
        'params': {
            'skip_connections': [11, 18, 38],
            'model_name': 'efficientnet-b7'
        }
    }
}
