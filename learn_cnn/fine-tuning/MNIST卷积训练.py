import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
"""
我们将加载MNIST数据集并进行预处理，将像素值缩放到0到1之间，并将数据集分为训练集和测试集。
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

num_classes = 10

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)
"""
接下来，我们将定义一个卷积神经网络模型。我们将使用两个卷积层和两个池化层，然后是两个全连接层和一个输出层。我们还将使用dropout和L2正则化来防止过拟合。
"""
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
"""
model.summary()是Keras中模型对象的一个方法，用于打印出模型的结构信息，包括每一层的名称、输出形状、参数数量等。这对于调试、优化模型以及理解模型结构都非常有用。
"""
model.summary()
"""
然后，我们将对模型进行编译，并使用数据增强技术来进一步防止过拟合。数据增强技术将应用一系列随机变换，例如旋转、平移、缩放等，来生成新的训练样本。这样可以使模型更加鲁棒，并防止过拟合。
"""
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
"""
接下来，我们将使用训练集来训练模型，并使用测试集来评估模型的性能。

"""
datagen.fit(x_train)
batch_size = 1024
epochs = 10
checkpoint = tf.keras.callbacks.ModelCheckpoint('./model.h5', save_best_only=True, save_weights_only=False, monitor='val_loss')
history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    steps_per_epoch=len(x_train) // batch_size,callbacks=[checkpoint])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])