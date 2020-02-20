import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.cv4 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
        )
        self.cv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.cv6 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.cv7 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
        )
        self.cv8 = nn.Sequential(
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.cv9 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.cv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.cv11 = nn.Sequential(
            nn.Conv2d(32, 4, 7, 1, 3),
            nn.ReLU(),
        )
        self.last = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*8*8, 100),
            nn.Tanh()
        )
        
    def forward(self, input):
        x = self.cv1(input)
        x = self.cv2(x)
        x = self.cv3(x)
        x = self.cv4(x)
        x = self.cv5(x)
        x = self.cv6(x)
        x = self.cv7(x)
        x = self.cv8(x)
        x = self.cv9(x)
        x = self.cv10(x)
        x = self.cv11(x)
        output = self.last(x)
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.ln = nn.Sequential(
            nn.Linear(100, 4*8*8),
            nn.ReLU()
        )
        self.cv1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
        )
        self.up1 =  nn.Upsample(size=(16, 16), mode='nearest')
        self.cv4 = nn.Sequential(
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU(),
        )
        self.up2 =  nn.Upsample(size=(32, 32), mode='nearest')
        self.cv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
        )
        self.cv6 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.up3 =  nn.Upsample(size=(64, 64), mode='nearest')
        self.cv7 = nn.Sequential(
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(),
        )
        self.up4 =  nn.Upsample(size=(64, 64), mode='nearest')
        self.cv8 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
        )
        self.up5 = nn.Upsample(size=(128, 128), mode='nearest')
        self.cv9 = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
        )
        self.up6 =  nn.Upsample(size=(256, 256), mode='nearest')
        self.cv10 = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(),
        )
        self.up7 =  nn.Upsample(size=(256, 256), mode='nearest')
        self.cv11 = nn.Sequential(
            nn.Conv2d(32, 3, 7, 1, 3),
            nn.Tanh(),
        )
        
    def forward(self, input):
        x = self.ln(input)
        x = x.view(-1,4,8,8)
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        x = self.up1(x)
        x = self.cv4(x)
        x = self.up2(x)
        x = self.cv5(x)
        x = self.cv6(x)
        x = self.up3(x)
        x = self.cv7(x)
        x = self.up4(x)
        x = self.cv8(x)
        x = self.up5(x)
        x = self.cv9(x)
        x = self.up6(x)
        x = self.cv10(x)
        x = self.up7(x)
        output = self.cv11(x)
        
        return output
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, input):
        code = self.encoder(input)
        output = self.decoder(code)
        
        return output