import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np;
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
# 加载模型并进行数字识别
def recognize_digit():
    # 将画板图像转换成灰度图像，并将其大小调整为 28x28,注意要用convert,因为彩色图像是rgb是三维的，resize只是改变了rg，需要convert转换成灰度的二维
    image_resized = np.array(image.resize((28, 28)).convert('L'))
    # 反转图像，因为灰度图像是黑底白字，但是我们训练的图片都是白底黑字，所以取反
    image_resized = np.invert(image_resized)
    # 将图像转换为数字数组
    data = image_resized.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    # 在这里添加您的识别代码
    model = tf.keras.models.load_model('./fine-tuning/model.h5')
    predictions = model.predict(data)
    result_label.configure(text="识别结果为：" + str(np.argmax(predictions)))

# 清空画板
def clear_canvas():
    draw.rectangle((0, 0, 280, 280), fill="white")
    canvas.delete("all")

# 创建窗口
window = tk.Tk()
window.title("卷积手写数字识别")
window.geometry("400x400")

# 创建画布
canvas = tk.Canvas(window, width=280, height=280, bg="white")
canvas.grid(row=0, column=0, columnspan=2)

# 创建清空画布按钮
clear_button = tk.Button(window, text="清空画板", command=clear_canvas)
clear_button.grid(row=1, column=0)

# 创建识别按钮
recognize_button = tk.Button(window, text="识别数字", command=recognize_digit)
recognize_button.grid(row=1, column=1)

# 创建识别结果标签
result_label = tk.Label(window, text="")
result_label.grid(row=2, column=0, columnspan=2)

# 创建画板图像
image = Image.new("RGB", (280, 280), (255, 255, 255))
draw = ImageDraw.Draw(image)

# 绑定画板事件
def on_mouse_down(event):
    global prev_x, prev_y
    prev_x, prev_y = event.x, event.y

def on_mouse_move(event):
    global prev_x, prev_y
    canvas.create_line(prev_x, prev_y, event.x, event.y, width=20)
    draw.line((prev_x, prev_y, event.x, event.y), fill="black", width=20)
    prev_x, prev_y = event.x, event.y

canvas.bind("<Button-1>", on_mouse_down)
canvas.bind("<B1-Motion>", on_mouse_move)

# 显示窗口
window.mainloop()