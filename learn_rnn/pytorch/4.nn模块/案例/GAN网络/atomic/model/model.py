import torch.nn as nn
"""
nn.ConvTranspose2d的参数包括：
    in_channels：输入通道数
    out_channels：输出通道数
    kernel_size：卷积核大小
    stride：步长
    padding：填充大小
    output_padding：输出填充大小
    groups：分组卷积数量，默认为1
    bias：是否使用偏置，默认为True
  生成器的目标是从一个随机噪声向量生成逼真的图像。在生成器中，通道数从大到小可以理解为从抽象的特征逐渐转化为具体的图像细节。通过逐层转置卷积（ConvTranspose2d）操作，
将低维度的特征逐渐转化为高维度的图像。通道数的减少可以理解为对特征进行提取和压缩，以生成更具细节和逼真度的图像。
"""
#生成网络
class Generator(nn.Module):
    def __init__(self, nz,ngf,nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # nz表示噪声的维度，一般是(100,1,1)
            # ngf表示生成特征图的维度
            # nc表示输入或者输出图像的维度
            #输出尺寸 = (输入尺寸（高度） - 1) * stride - 2 * padding + kernel_size + output_padding
            #如果（卷积核,步长，填充）=(4, 1, 0)表示图像的维度是卷积核的大小（卷积核高,卷积核宽）
            #如果（卷积核,步长，填充）=(4, 2, 1)表示图像的维度是是上一个图像的2被（输入图像高度*2,输入图像宽度*2）
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

"""
    和转置卷积相反的是（4,2,1）会让维度2倍降低
    卷积过程是height-kerel+1
"""
class Discriminator(nn.Module):
    def __init__(self, nc,ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size （1,1,1）
        )

    def forward(self, input):
        return self.main(input)