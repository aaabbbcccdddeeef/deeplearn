import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential

np_config.enable_numpy_behavior()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def cifar10_generator(x, y, batch_size):
    while True:
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            x_batch = tf.image.resize_with_pad(x_batch, target_height=224, target_width=224)
            x_batch = x_batch.astype('float32') / 255.0
            yield x_batch, y_batch


def vggnet(input_shape, num_classes):
    # 定义VGGNet
    model = Sequential([
        # 第一层卷积和池化
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # 第二层卷积和池化
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # 第三层卷积和池化
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # 第四层卷积和池化
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # 第五层卷积和池化
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),

        # 将输出的特征图展平，并连接全连接层
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(10, activation='softmax')
    ])

    return model

# 定义一些超参数
batch_size = 128
epochs = 5
learning_rate = 0.001

# 定义生成器
train_generator = cifar10_generator(x_train, y_train, batch_size)
test_generator = cifar10_generator(x_test, y_test, batch_size)

# 定义模型
input_shape = (224,224,3)
num_classes = 10
model = vggnet(input_shape, num_classes)
model.summary()
# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 定义 ModelCheckpoint 回调函数
checkpoint = tf.keras.callbacks.ModelCheckpoint('./VGGNet.h5', save_best_only=True, save_weights_only=False, monitor='val_loss')

# 训练模型
model.fit(train_generator,
          epochs=epochs,
          steps_per_epoch=len(x_train)//batch_size,
          validation_data=test_generator,
          validation_steps=len(x_test)//batch_size,
          callbacks=[checkpoint]
          )
test_loss, test_acc = model.evaluate(test_generator, y_test)
print('Test accuracy:', test_acc)


