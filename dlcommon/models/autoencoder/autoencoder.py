import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, channel, enc_channel, n_res_block, n_res_channel):
        super().__init__()

        blocks = [
            nn.Conv2d(in_channel, channel // 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 32, channel // 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel // 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel // 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, enc_channel, 3, padding=1),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(enc_channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
        
class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, in_channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(in_channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        blocks.extend(
            [
                nn.ConvTranspose2d(in_channel, channel, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, channel // 4, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 4, channel // 8, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 8, channel // 16, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 16, channel // 32, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    channel // 32, out_channel, 4, stride=2, padding=1
                ),
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)
    
class AutoEncoder(nn.Module):
    def __init__(self,
        in_channel=3,
        channel=256,
        enc_channel = 32,
        n_res_block=2,
        n_res_channel=8,
    ):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_channel, channel, enc_channel, n_res_block, n_res_channel)
        self.decoder = Decoder(enc_channel, in_channel, channel, n_res_block, n_res_channel)
        
    def forward(self, input):
        code = self.encoder(input)
        output = self.decoder(code)
        
        return output