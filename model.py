import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_loader import data_loader
from utility import speakers


class Down2d(nn.Module):
    """docstring for Down2d."""
    def __init__(self, in_channel ,out_channel, kernel, stride, padding):
        super(Down2d, self).__init__()

        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)
    
    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)

        x2 = self.c2(x)
        x2 = self.n2(x2)

        x3 =  x1 * torch.sigmoid(x2)
        
        return x3
            
        
class Up2d(nn.Module):
    """docstring for Up2d."""
    def __init__(self, in_channel ,out_channel, kernel, stride, padding):
        super(Up2d, self).__init__()
        self.c1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)
         
    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)

        x2 = self.c2(x)
        x2 = self.n2(x2)

        x3 =  x1 * torch.sigmoid(x2)
        
        return x3
        

class Generator(nn.Module):
    """docstring for Generator."""
    def __init__(self):
        super(Generator, self).__init__()
        self.downsample = nn.Sequential(
            Down2d(1, 32, (3,9), (1,1), (1,4)),
            Down2d(32, 64, (4,8), (2,2), (1,3)),
            Down2d(64, 128, (4,8), (2,2), (1,3)),
            Down2d(128, 64, (3,5), (1,1), (1,2)),
            Down2d(64, 5, (9,5), (9,1), (1,2))
        )

        
        self.up1 = Up2d(9, 64, (9,5), (9,1), (0,2))
        self.up2 = Up2d(68, 128, (3,5), (1,1), (1,2))
        self.up3 = Up2d(132, 64, (4,8), (2,2), (1,3))
        self.up4 = Up2d(68, 32, (4,8), (2,2), (1,3))

        self.deconv = nn.ConvTranspose2d(36, 1, (3,9), (1,1), (1,4))

    def forward(self, x, c):
        x = self.downsample(x)
        c = c.view(c.size(0), c.size(1), 1, 1)

        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        x = self.up1(x)

        c2 = c.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.up2(x)

        c3 = c.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.up3(x)

        c4 = c.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.up4(x)

        c5 = c.repeat(1,1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    """docstring for Discriminator."""
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.d1 = Down2d(5, 32, (3,9), (1,1), (1,4))
        self.d2 = Down2d(36, 32, (3,8), (1,2), (1,3))    
        self.d3 = Down2d(36, 32, (3,8), (1,2), (1,3))    
        self.d4 = Down2d(36, 32, (3,6), (1,2), (1,2)) 
        
        self.conv = nn.Conv2d(36, 1, (36,5), (36,1), (0,2))
        self.pool = nn.AvgPool2d((1,64))
    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        
        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        x = self.d1(x)

        c2 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.d2(x)

        c3 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.d3(x)

        c4 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.d4(x)

        c5 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.conv(x)
       
        x = self.pool(x)
        x = torch.squeeze(x)
        x = torch.tanh(x)
        return x

class DomainClassifier(nn.Module):
    """docstring for DomainClassifier."""
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.main = nn.Sequential(
            Down2d(1, 8, (4,4), (2,2), (5,1)),
            Down2d(8, 16, (4,4), (2,2), (1,1)),
            Down2d(16, 32, (4,4), (2,2), (0,1)),
            Down2d(32, 16, (3,4), (1,2), (1,1)),
            nn.Conv2d(16, 4, (1,4), (1,2), (0,1)),
            nn.AvgPool2d((1,16)),
            nn.LogSoftmax()
        )
        
    def forward(self, x):
       x = x[:, :, 0:8, :]
       x = self.main(x)
       x = x.view(x.size(0), x.size(1))
       return x



if __name__ == '__main__':
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # train_loader = data_loader('data/processed', 1)
    # data_iter = iter(train_loader)


    t = torch.rand([1,1,36,512])
    l = torch.FloatTensor([[0,1,0,0]])
    print(l.size())
    # d1 = Down2d(1, 32, 1,1,1)
    # print(d1(t).shape)

    # u1 = Up2d(1,32,1,2,1)
    # print(u1(t).shape)
    # G = Generator()
    # o1 = G(t, l)
    # print(o1.shape)
    
    D = Discriminator()
    o2 = D(t, l)
    print(o2.shape, o2)

    # C = DomainClassifier()
    # o3 = C(t)
    # print(o3.shape)
    # m = nn.Softmax()
    # input = torch.Tensor([[1,2],[5,5]])
    # output = m(input)
    # print(output)