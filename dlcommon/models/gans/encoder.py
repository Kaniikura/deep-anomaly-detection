import numpy as np
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

def make_block(dim, n_block):
    blocks = []
    curr_dim =dim
    for i in range(n_block):
        blocks.extend([
                  nn.ReLU(),
                  nn.BatchNorm2d(curr_dim),
                  nn.Conv2d(curr_dim, curr_dim * 2, 4, stride=2, padding=1),
        ])
        curr_dim = curr_dim * 2

    return blocks

class ResEncoder(nn.Module):
    def __init__(self, image_size=128, z_dim=100, max_dim=512, n_res_block=2):
        super().__init__()
        repeat_num = int(np.log2(image_size)) - 2
        mult = 2 ** repeat_num # img_size: 32 -> 8, img_size: 128 -> 32

        blocks = [
            nn.Conv2d(3, max_dim // mult, 4, stride=2, padding=1),
        ]

        curr_dim = max_dim // mult

        if image_size==128:
          blocks.extend(make_block(curr_dim, 2))
          curr_dim = curr_dim * 4
        else:
          blocks.extend(make_block(curr_dim, 1))
          curr_dim = curr_dim * 2

        for i in range(n_res_block):
            blocks.append(ResBlock(curr_dim, curr_dim // 2))

        if image_size==128:
          blocks.extend(make_block(curr_dim, 2))
          curr_dim = curr_dim * 4
        else:
          blocks.extend(make_block(curr_dim, 1))
          curr_dim = curr_dim * 2


        for i in range(n_res_block):
            blocks.append(ResBlock(curr_dim, curr_dim // 2))

        blocks.extend([
            nn.ReLU(),
            nn.BatchNorm2d(max_dim // 2),
            nn.Conv2d(max_dim // 2, max_dim,  4, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(max_dim),
            nn.Conv2d(max_dim, max_dim, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(max_dim),
            
        ])

        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_dim, z_dim),
            nn.ReLU(),
            nn.BatchNorm1d(z_dim),
        )

    def forward(self, input):
        x = self.blocks(input)
        x = self.linear(x)
        return x