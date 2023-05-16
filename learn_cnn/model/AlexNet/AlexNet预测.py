import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

np_config.enable_numpy_behavior()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# 在这里添加您的识别代码
model = tf.keras.models.load_model('./AlexNet.h5')

srcImage=x_test[100]
p_test=np.array([srcImage])
p_test = tf.image.resize_with_pad(p_test, target_height=224, target_width=224)
p_test = p_test.astype('float32') / 255.0
predictions = model.predict(p_test)
print("识别结果为：" + str(np.argmax(predictions)))
# 绘制第10个测试数据的图形
plt.imshow(x_test[100], cmap=plt.cm.binary)
plt.show()