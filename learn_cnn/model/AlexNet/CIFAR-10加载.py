import matplotlib.pyplot as plt
from keras.datasets import cifar10
import numpy as np
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)
"""
执行后，日志里有一直在下载的过程，下载很慢，路径
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
我们可以手动下载下来，重命名为：cifar-10-batches-py.tar.gz，然后上传到 ~/.keras/datasets目录即可（不用解压），程序会离线解压该文件，window下是：C:\\Users\\你的用户\\.keras\\datasets
"""
# 随机选择100张图片进行显示
indices = np.random.choice(len(x_train), size=100, replace=False)
images = x_train[indices]
labels = y_train[indices]

# 绘制图片
fig = plt.figure(figsize=(10, 10))
for i in range(10):
    for j in range(10):
        index = i * 10 + j
        ax = fig.add_subplot(10, 10, index + 1)
        ax.imshow(images[index])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(labels[index][0])
plt.show()
