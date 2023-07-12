import fire
import torch
from config import Config
from data.dataset import AtomicDataset
from model.model import Generator, Discriminator
import torch.nn as nn
from matplotlib import pyplot as plt
import torchvision.utils as vutils
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML
# 判断使用的设备
from torch import optim

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


def train(**kwargs):
    # 通过传入的参数初始化Config
    defaultConfig = Config(kwargs)
    # 通过给定的目录和图像大小转换成数据集
    dataset = AtomicDataset(defaultConfig.img_root, defaultConfig.img_size)
    # 转换为可迭代的批次为defaultConfig.batch_size的数据集
    dataloader = dataset.toBatchLoader(defaultConfig.batch_size)
    # 创建生成网络模型
    netG = Generator(defaultConfig.nz, defaultConfig.ngf, defaultConfig.nc).to(device)
    # 创建分类器模型
    netD = Discriminator(defaultConfig.nc, defaultConfig.ndf).to(device)
    # 使用criterion = nn.BCELoss()
    criterion = nn.BCELoss()
    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=defaultConfig.lr, betas=(defaultConfig.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=defaultConfig.lr, betas=(defaultConfig.beta1, 0.999))
    # 如果是真的图片label=1，伪造的图片为0
    real_label = 1
    fake_label = 0
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    #生成一个64批次100*1*1的噪声
    fixed_noise = torch.randn(64, defaultConfig.nz, 1, 1, device=device)

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(defaultConfig.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # 对于真实传入的图片进行判断器训练，label肯定是1
            # 对于噪声传入的图片进行判断器训练，label肯定是0
            ###########################
            ## 通过真实图片训练D网络
            netD.zero_grad()
            # 将64批次数据转换为gpu设备
            real_cpu = data[0].to(device)
            # 获取批次的个数
            b_size = real_cpu.size(0)
            # 生成的是一个一维的张量，其中包含64个元素,每个元素的值为1。
            label = torch.full((b_size,), real_label, device=device).float()
            # 分类器捲積后最后产生一个64个批次的1*1，转换成1维数组。
            output = netD(real_cpu).view(-1)
            # 计算和真实数据的损失
            errD_real = criterion(output, label)
            # 反向传播计算梯度
            errD_real.backward()
            # D_x的值表示判别器对真实样本的平均预测概率
            D_x = output.mean().item()

            ## 通过噪声训练生成器模型
            # 生成噪声的变量 也是64批次，噪声的通道数是100
            noise = torch.randn(b_size, defaultConfig.nz, 1, 1, device=device)
            # 传入到生成网络中，生成一张64*3*64*64的图片
            fake = netG(noise)
            # 生成器生成的图片对应的真实的label应该是0
            label.fill_(fake_label)
            # detach()是PyTorch中的一个函数，它用于从计算图中分离出一个Tensor。当我们调用detach()函数时，它会返回一个新的Tensor，该Tensor与原始Tensor共享相同的底层数据，但不会有梯度信息。
            # 使用判别器网络来判断通过噪声生成的图片，转换为1维
            output = netD(fake.detach()).view(-1)
            # 进行损失函数计算
            errD_fake = criterion(output, label)
            # 反向传播计算梯度
            errD_fake.backward()
            # 表示判别器对虚假样本的平均预测概率
            D_G_z1 = output.mean().item()
            # 将真实图片和虚假图片的损失求和获取所有的损失
            errD = errD_real + errD_fake
            # 更新权重参数
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            # 对于G网络来说，对于虚假传入的图片进行判断器训练，尽量让判别器认为是真1，生成的图片才够真实
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # 使用之前的G网络生成的图片64*3*64*64,传入D网络
            output = netD(fake).view(-1)
            # 计算G网路的损失
            errG = criterion(output, label)
            # 反向计算梯度
            errG.backward()
            #表示判别器对虚假样本判断为真的的平均预测概率
            D_G_z2 = output.mean().item()
            # 更新G的权重
            optimizerG.step()

            # 输出训练统计，每1000批次
            if i % 1000 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, defaultConfig.num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # 即每经过一定数量的迭代（iters % 250 == 0）或者是训练的最后一个epoch的最后一个batch（(epoch == defaultConfig.num_epochs - 1) and (i == len(dataloader) - 1)），
            # 就会使用G网络通过噪声生成64批次3通道64*64的图像，并且加入到img_list去做可视化，看看效果
            if (iters % 250 == 0) or ((epoch == defaultConfig.num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    #保存生成器的网络到checkpoints目录
    torch.save(netG.state_dict(), "./checkpoints/optimizerG.pt")
    #绘制G和D的损失函数图像
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    #一维数组的索引值是x坐标也就是批次索引
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    #创建一个8*8的画布
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())
if __name__ == "__main__":
    # 将main.py中所有的函数映射成  python main.py 方法名 --参数1=参数值 --参数2=参数值的形式，这些参数以keyvalue字典的形式传入kwargs
    fire.Fire()
