#%%

import os
import numpy as np
import tensorflow.keras as layers
import tensorflow as tf
import datetime

TARGET_MODEL_DIR="./"
MODEL_NAME="model.h5"
epochs_count=5
"""
   jupyter打印的日志太大导致ipynb打开很慢，这里写个一模一样代码的py运行
"""
def againTrain(x_train, y_train, x_test, y_test):
    targetModel=os.path.join(TARGET_MODEL_DIR,MODEL_NAME)
    #记载CNN模型
    model=tf.keras.models.load_model(targetModel)
    """
    在使用Fine-tuning方法微调预训练模型时，通常会冻结模型的前几层，只调整模型的后面几层，这是因为：
    1.预训练模型的前几层通常是针对原始数据集的通用特征提取器，这些特征对于不同的任务和数据集都是有用的，因此我们可以直接保留这些特征提取器，不需要进行微调。
    2.预训练模型的后几层通常是针对特定任务进行的微调，这些层的参数需要根据具体任务和数据集进行调整，以使模型更好地适应特定的任务和数据集。
    3.如果我们将整个模型的所有层都进行微调，会导致训练时间较长，而且可能会出现过拟合等问题。因此，冻结前几层可以有效地减少训练时间，并提高模型的泛化能力。
    总之，冻结模型的前几层可以节省计算资源和训练时间，同时还可以提高模型的泛化能力，使其更好地适应新的任务和数据集。
    """
    model.layers[0].trainable = False
    model.layers[1].trainable = False
    # 对输入图像进行预处理
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    x_test = x_test.astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    now = datetime.datetime.now()  # 获取当前时间
    format_time = now.strftime("%Y-%m-%d%H-%M-%S")  # 转换为指定格式
    checkpoint = tf.keras.callbacks.ModelCheckpoint(targetModel, save_best_only=True, save_weights_only=False, monitor='val_loss')
    # 继续训练模型
    history = model.fit(x_train, y_train, batch_size=128, epochs=epochs_count, validation_data=(x_test, y_test),
                        callbacks=[checkpoint])
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
"""
  传入mnist-c，数据会非常大加载数据很慢
"""
def loadDataMnistC(data_root,func):
    dirlist=os.listdir(data_root)
    for i, folder_name in enumerate(dirlist):
        folder_path = os.path.join(data_root, folder_name)
        if os.path.isdir(folder_path):
            print("开始读取："+folder_path)
            train_images = np.load(os.path.join(folder_path, 'train_images.npy'))
            train_labels = np.load(os.path.join(folder_path, 'train_labels.npy'))
            test_images = np.load(os.path.join(folder_path, 'test_images.npy'))
            test_labels = np.load(os.path.join(folder_path, 'test_labels.npy'))
            print("开始训练："+folder_path)
            func(train_images,train_labels,test_images,test_labels)
            print("训练完成："+folder_path)
# 加载 MNIST-C 数据集
data_root = './mnist_c'
model=None;
loadDataMnistC(data_root,againTrain)
print("全部训练完成")


