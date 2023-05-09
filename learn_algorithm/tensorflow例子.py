#%%
import numpy as np;

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([1, 2, 3])
print(sigmoid(x))
# 输出：[0.73105858 0.88079708 0.95257413]


import tensorflow as tf

# 创建一个常量张量
x = tf.constant([1, 2, 3])

# 创建一个变量张量
y = tf.Variable([3, 2, 1])

# 计算两个张量的和
z = tf.add(x, y)

# 输出结果
print(z.numpy())

