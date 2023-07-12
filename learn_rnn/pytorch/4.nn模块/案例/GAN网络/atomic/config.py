class Config:
    #定义转换后图像的大小
    img_size=64
    #训练图片所在目录，目录必须是有子目录，子目录名称就是分类名
    img_root="./data/AnimeFaceDataset"
    #每次加载的批次数
    batch_size=64
    """
    在卷积神经网络中，这些缩写通常表示以下含义：
        nz：表示输入噪声向量的维度。全称为"noise dimension"，即噪声维度。
        ngf：表示生成器网络中特征图的通道数。全称为"number of generator features"，即生成器特征图通道数。
        nc：表示输入图像的通道数。全称为"number of image channels"，即图像通道数。
    """
    #表示噪声的维度，一般是(100,1,1)
    nz=100
    #表示生成特征图的维度,64*64的图片
    ngf=64
    #生成或者传入图片的通道数
    nc=3
    # 表示判别器输入特征图的维度,64*64的图片
    ndf = 64
    # 优化器的学习率
    lr = 0.0002
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    # epochs的次数
    num_epochs=50
    def __init__(self,kv):
        for key, value in kv.items():
            setattr(self, key, value)
