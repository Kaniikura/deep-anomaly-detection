# https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import SpectralNorm
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out#,attention

class Generator(nn.Module):
    """Generator."""

    def __init__(self, image_size=128, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # img_size: 32 -> 4, img_size: 128 -> 16
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        if self.imsize == 128:
            self.attn1 = Self_Attn( 256, 'relu')
            self.attn2 = Self_Attn( 128,  'relu')
            self.attn3 = Self_Attn( 64,  'relu')

            layer4 = []
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

            layer5 = []
            layer5.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer5.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer5.append(nn.ReLU())
            self.l5 = nn.Sequential(*layer5)
            curr_dim = int(curr_dim / 2)

            self.sa_conv = nn.Sequential(self.l1, self.l2, self.l3, self.attn1,
                                         self.l4, self.attn2, self.l5, self.attn3)

        else:
            self.attn1 = Self_Attn( 64, 'relu')
            self.sa_conv = nn.Sequential(self.l1, self.l2, self.l3, self.attn1)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        x = self.sa_conv(z)
        out = self.last(x)

        return out


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, image_size=128, z_dim=100, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        if self.imsize == 128:
            self.attn1 = Self_Attn(256, 'relu')
            self.attn2 = Self_Attn(512, 'relu')
            self.attn3 = Self_Attn(1024, 'relu')

            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2

            layer5 = []
            layer5.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer5.append(nn.LeakyReLU(0.1))
            self.l5 = nn.Sequential(*layer5)
            curr_dim = curr_dim*2

            self.sa_conv = nn.Sequential(self.l1, self.l2, self.l3, self.attn1,
                                         self.l4, self.attn2, self.l5, self.attn3)
        
        else:
            self.attn1 = Self_Attn(256, 'relu')
            self.sa_conv = nn.Sequential(self.l1, self.l2, self.l3, self.attn1)
        
        self.last_feature = nn.Sequential(
            nn.Conv2d(curr_dim, z_dim, 4),
            nn.Flatten(),
        )

        self.last_ln = nn.Linear(z_dim, 1)

    def get_feature(self, x):
        x = self.sa_conv(x)
        out = self.last_feature(x)

        return out

    def forward(self, x):
        x = self.get_feature(x)
        out = self.last_ln(x)

        return out

class SAGAN():
    def __init__(self, image_size=128, z_dim=100, conv_dim=64):
        self.G = Generator(image_size, z_dim, conv_dim)
        self.D = Discriminator(image_size, z_dim, conv_dim)
