import numpy as np
import torch.nn as nn
from custom.activations import Mish

class ResBlock(nn.Module):
    def __init__(self, in_max_dim, max_dim):
        super().__init__()

        self.conv = nn.Sequential(
            Mish(),
            nn.BatchNorm2d(in_max_dim),
            nn.Conv2d(in_max_dim, max_dim, 3, padding=1),
            Mish(),
            nn.BatchNorm2d(max_dim),
            nn.Conv2d(max_dim, in_max_dim, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

def downsample_block(dim, n_block):
    blocks = []
    curr_dim =dim
    for i in range(n_block):
        blocks.extend([
                  Mish(),
                  nn.BatchNorm2d(curr_dim),
                  nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1),
        ])
        curr_dim = curr_dim * 2

    return blocks

def upsample_block(dim, n_block):
    blocks = []
    curr_dim =dim
    for i in range(n_block):
        blocks.extend([
                  Mish(),
                  nn.BatchNorm2d(curr_dim),
                  nn.ConvTranspose2d(curr_dim, curr_dim // 2, 4, stride=2, padding=1),   
        ])
        curr_dim = curr_dim // 2

    return blocks

class Encoder(nn.Module):
    def __init__(self, image_size=128, z_dim=100, max_dim=512, n_res_block=2):
        super().__init__()
        repeat_num = int(np.log2(image_size)) - 2
        mult = 2 ** repeat_num # img_size: 32 -> 8, img_size: 128 -> 32

        blocks = [
            nn.Conv2d(3, max_dim // mult, 4, stride=2, padding=1),
        ]

        curr_dim = max_dim // mult

        if image_size==256:
          blocks.extend(downsample_block(curr_dim, 3))
          curr_dim = curr_dim * 8
        else:
          blocks.extend(downsample_block(curr_dim, 1))
          curr_dim = curr_dim * 2

        for i in range(n_res_block):
            blocks.append(ResBlock(curr_dim, curr_dim // 2))

        if image_size==256:
          blocks.extend(downsample_block(curr_dim, 2))
          curr_dim = curr_dim * 4
        else:
          blocks.extend(downsample_block(curr_dim, 1))
          curr_dim = curr_dim * 2


        for i in range(n_res_block):
            blocks.append(ResBlock(curr_dim, curr_dim // 2))

        blocks.extend([
            Mish(),
            nn.BatchNorm2d(max_dim // 2),
            nn.Conv2d(max_dim // 2, max_dim,  4, stride=1),
            Mish(),
            nn.BatchNorm2d(max_dim), 
        ])

        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_dim, z_dim),
            Mish(),
            nn.BatchNorm1d(z_dim),
        )

    def forward(self, input):
        x = self.blocks(input)
        x = self.linear(x)
        return x
        
        
class Decoder(nn.Module):
    def __init__(self, image_size=128, z_dim=100, max_dim=512, n_res_block=2):
        super().__init__()
        self.max_dim = max_dim

        self.linear = nn.Sequential(
            nn.Linear(z_dim, max_dim),
            Mish(),
            nn.BatchNorm1d(max_dim),
        )

        curr_dim = max_dim

        blocks = [
            nn.ConvTranspose2d(max_dim, max_dim // 2, 4, stride=1),
            Mish(),
            nn.BatchNorm2d(max_dim // 2),
        ]

        curr_dim = curr_dim // 2

        for i in range(n_res_block):
            blocks.append(ResBlock(curr_dim, curr_dim // 2))

        if image_size==256:
            blocks.extend(upsample_block(curr_dim, 2))
            curr_dim = curr_dim // 4
        else:
            blocks.extend(upsample_block(curr_dim, 1))
            curr_dim = curr_dim // 2
          
        for i in range(n_res_block):
            blocks.append(ResBlock(curr_dim, curr_dim // 2))

        if image_size==256:
            blocks.extend(upsample_block(curr_dim, 3))
            curr_dim = curr_dim // 8
        else:
            blocks.extend(upsample_block(curr_dim, 1))
            curr_dim = curr_dim // 2

        for i in range(n_res_block):
            blocks.append(ResBlock(curr_dim, curr_dim // 2))

        blocks.extend([
            nn.ConvTranspose2d(curr_dim, 3, 4, stride=2, padding=1),
        ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.linear(input)
        x = x.view(-1, self.max_dim, 1, 1)
        x = self.blocks(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self,
        image_size=256,
        z_dim=100,
        max_dim=512, 
        n_res_block=2
    ):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(image_size, z_dim, max_dim, n_res_block)
        self.decoder = Decoder(image_size, z_dim, max_dim, n_res_block)
        
    def forward(self, input):
        code = self.encoder(input)
        output = self.decoder(code)
        
        return output