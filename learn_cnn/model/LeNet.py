import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
#开启tensorflow支持numpy函数，astype是numpy的函数
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
ori_x_test1=x_test

# 将图像从28*28转换成32*32
x_train = tf.pad(x_train, [[0,0], [2,2], [2,2]], mode='constant')
x_test = tf.pad(x_test, [[0,0], [2,2], [2,2]], mode='constant')

# 将像素值缩放到0-1之间
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

# 定义Lenet-5模型
model = models.Sequential([
    # 第一层卷积层，6个卷积核，大小为5*5，使用sigmoid激活函数
    layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)),
    # 第一层池化层，大小为2*2
    layers.MaxPooling2D((2, 2)),
    # 第二层卷积层，16个卷积核，大小为5*5，使用sigmoid激活函数
    layers.Conv2D(16, (5, 5), activation='relu'),
    # 第二层池化层，大小为2*2
    layers.MaxPooling2D((2, 2)),
    # 第三层卷积层，120个卷积核，大小为5*5，使用sigmoid激活函数
    layers.Conv2D(120, (5, 5), activation='relu'),
    # 将卷积层的输出拉平
    layers.Flatten(),
    # 第一层全连接层，84个节点，使用sigmoid激活函数
    layers.Dense(84, activation='relu'),
    # 输出层，共10个节点，对应0-9十个数字，使用softmax激活函数
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



#取出其中一个测试数据进行测试
testdata = ori_x_test1[100]
testdata = testdata.reshape(-1,28,28)
testdata = tf.pad(testdata, [[0,0], [2,2], [2,2]], mode='constant')
testdata=testdata.reshape(-1, 32, 32, 1)
# 将像素值缩放到0-1之间
testdata = testdata.astype('float32') / 255.0
predictions = model.predict(testdata)
print("预测结果：", np.argmax(predictions))

# 绘制第10个测试数据的图形
plt.imshow(ori_x_test1[100], cmap=plt.cm.binary)
plt.show()