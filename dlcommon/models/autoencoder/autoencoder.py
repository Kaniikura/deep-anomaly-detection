import torch.nn as nn
from custom.activations import Mish

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            Mish(),
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            Mish(),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, z_dim, channel, n_res_block):
        super().__init__()
        self.channel = channel

        blocks = [
            nn.Conv2d(in_channel, channel // 64, 4, stride=2, padding=1),
            Mish(),
            nn.BatchNorm2d(channel // 64),
            nn.Conv2d(channel // 64, channel // 32, 4, stride=2, padding=1),
            Mish(),
            nn.BatchNorm2d(channel // 32),
            nn.Conv2d(channel // 32, channel // 16, 4, stride=2, padding=1),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel // 16, channel // 32))

        blocks.extend([
            Mish(),
            nn.BatchNorm2d(channel // 16),
            nn.Conv2d(channel // 16, channel // 8, 4, stride=2, padding=1),
            Mish(),
            nn.BatchNorm2d(channel // 8),
            nn.Conv2d(channel // 8, channel // 4, 4, stride=2, padding=1),
        ])

        for i in range(n_res_block):
            blocks.append(ResBlock(channel // 4, channel // 8))

        blocks.extend([
            Mish(),
            nn.BatchNorm2d(channel // 4),
            nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1),
            Mish(),
            nn.BatchNorm2d(channel // 2),
            nn.Conv2d(channel // 2, channel,  4, stride=1),
            Mish(),
            nn.BatchNorm2d(channel),
            nn.Conv2d(channel, channel, 3, padding=1),
            Mish(),
            nn.BatchNorm2d(channel),
            
        ])

        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel, z_dim),
            Mish(),
            nn.BatchNorm1d(z_dim),
        )

    def forward(self, input):
        x = self.blocks(input)
        x = self.linear(x)
        return x
        
class Decoder(nn.Module):
    def __init__(self, out_channel, z_dim, channel, n_res_block):
        super().__init__()
        self.channel = channel

        self.linear = nn.Sequential(
            nn.Linear(z_dim, channel),
            Mish(),
            nn.BatchNorm1d(channel),
        )

        blocks = [
            nn.ConvTranspose2d(channel, channel // 2, 4, stride=1),
            Mish(),
            nn.BatchNorm2d(channel // 2),
            nn.ConvTranspose2d(channel // 2, channel // 4, 4, stride=2, padding=1),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel // 4, channel // 8))

        blocks.extend([
            Mish(),
            nn.BatchNorm2d(channel // 4),
            nn.ConvTranspose2d(channel // 4, channel // 8, 4, stride=2, padding=1),
            Mish(),
            nn.BatchNorm2d(channel // 8),
            nn.ConvTranspose2d(channel // 8, channel // 16, 4, stride=2, padding=1),
        ])

        for i in range(n_res_block):
            blocks.append(ResBlock(channel // 16, channel // 32))

        blocks.extend(
            [
                Mish(),
                nn.BatchNorm2d(channel // 16),
                nn.ConvTranspose2d(channel // 16, channel // 32, 4, stride=2, padding=1),
                Mish(),
                nn.BatchNorm2d(channel // 32),
                nn.ConvTranspose2d(channel // 32, channel // 64, 4, stride=2, padding=1),
                Mish(),
                nn.BatchNorm2d(channel // 64),
                nn.ConvTranspose2d(
                    channel // 64, out_channel, 4, stride=2, padding=1
                ),
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        x = self.linear(input)
        x = x.view(-1, self.channel, 1, 1)
        x = self.blocks(x)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self,
        in_channel=3,
        z_dim = 100,
        channel=256,
        n_res_block=2,
    ):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_channel, z_dim, channel, n_res_block)
        self.decoder = Decoder(in_channel, z_dim, channel, n_res_block)
        
    def forward(self, input):
        code = self.encoder(input)
        output = self.decoder(code)
        
        return output