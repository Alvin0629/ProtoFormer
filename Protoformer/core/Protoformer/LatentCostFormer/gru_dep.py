import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3x3(nn.Module): 
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
    
class ConvBlock(nn.Module):  
    """Layer to perform a convolution followed by LeakyReLU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class DepthHead(nn.Module): 
    def __init__(self):
        super(DepthHead, self).__init__()
        self.covd1 = torch.nn.Sequential(nn.ReflectionPad2d(1),
                                         torch.nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1,
                                                         padding=0, bias=True),
                                         torch.nn.LeakyReLU(inplace=True))
        self.covd2 = torch.nn.Sequential(nn.ReflectionPad2d(1),
                                         torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1,  # out channel is 1
                                                         padding=0, bias=True))

    def forward(self, x):
        return self.covd2(self.covd1(x))


class SepConvGRU(nn.Module):
    def __init__(self):
        super(SepConvGRU, self).__init__()
        hidden_dim = 128
        catt = 256

        self.convz1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))
        self.convr1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))
        self.convq1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))

        self.convz2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))
        self.convr2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))
        self.convq2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h

class BasicMotionEncoder(nn.Module): 
    def __init__(self):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = ConvBlock(128, 160)
        self.convc2 = ConvBlock(160, 128)
        self.convf1 = torch.nn.Sequential(
            nn.ReflectionPad2d(3),
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True))
        self.convf2 = torch.nn.Sequential(
            nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=0, bias=True),
            torch.nn.LeakyReLU(inplace=True))

        self.conv = ConvBlock(128 + 32, 192 - 1)

    def forward(self, depth, corr):
        cor = self.convc1(corr)
        cor = self.convc2(cor)
        dep = self.convf1(depth)
        dep = self.convf2(dep)
        cor_depth = torch.cat([cor, dep], dim=1)
        out = self.conv(cor_depth)
        return torch.cat([out, depth], dim=1)
    

from .gma import Aggregate
class GMAUpdateBlock(nn.Module):            
    def __init__(self, args, hidden_dim=128):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder()
        self.depth_head = DepthHead()  

        self.mask = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(192, 324, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(324, 64 * 9, 1, padding=0))

        self.aggregator = Aggregate(args=self.args, dim=126, dim_head=128, heads=1)  

        

    def forward(self, net, corr, depth):
        net = self.encoder(depth, corr)
        
        # Depth head
        delta_depth = self.depth_head(net)
        
        # scale mask
        mask = .25 * self.mask(net)
        return net, mask, delta_depth
