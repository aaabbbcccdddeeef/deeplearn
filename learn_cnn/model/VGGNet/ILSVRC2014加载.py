from keras.datasets import ilsvrc2014

# 加载ILSVRC2014训练集
(x_train, y_train), (_, _) = ilsvrc2014.load_data(train=True)

# 加载ILSVRC2014测试集
(x_test, y_test), (_, _) = ilsvrc2014.load_data(train=False)

# 打印训练集和测试集的形状和标签数目
print('训练集形状:', x_train.shape)
print('训练集标签数目:', len(set(y_train)))
print('测试集形状:', x_test.shape)
print('测试集标签数目:', len(set(y_test)))