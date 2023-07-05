# RNN
`nn.RNN`是PyTorch中的一个循环神经网络模块，用于处理序列数据。下面是`nn.RNN`的常用参数和解释：

*   `input_size`：输入的特征维度。
*   `hidden_size`：隐藏层的特征维度。
*   `num_layers`：RNN的层数。
*   `nonlinearity`：激活函数，默认为"tanh"。可以是"tanh"、"relu"等。
*   `bias`：是否使用偏置，默认为True。
*   `batch_first`：是否输入数据的第一个维度为batch大小，默认为False。
*   `dropout`：是否在输出层应用dropout操作，默认为0，即不使用dropout。
*   `bidirectional`：是否使用双向RNN，默认为False。

这些参数可以在创建`nn.RNN`时进行设置。例如：

```import torch
import torch.nn as nn

input_size = 10
hidden_size = 20
num_layers = 2

rnn = nn.RNN(input_size, hidden_size, num_layers)
```

这样就创建了一个具有输入特征维度为10、隐藏层特征维度为20、2层的RNN模型。
## 传入数据格式
`nn.RNN`的输入数据格式通常为三维张量，具体格式为：

*   如果`batch_first=False`（默认值），则输入数据的形状为`(sequence_length, batch_size, input_size)`。
*   如果`batch_first=True`，则输入数据的形状为`(batch_size, sequence_length, input_size)`。

其中，

*   `sequence_length`表示序列的长度，即时间步的数目。
*   `batch_size`表示每个batch的样本数量。
*   `input_size`表示输入特征的维度。

例如，假设我们有一个batch包含3个样本，每个样本的序列长度为4，输入特征维度为5，那么输入数据的形状可以是`(4, 3, 5)`或`(3, 4, 5)`。

可以使用`torch.randn()`函数生成随机输入数据进行测试，例如：

`import torch
import torch.nn as nn

batch_size = 3
sequence_length = 4
input_size = 5

input_data = torch.randn(sequence_length, batch_size, input_size)
rnn = nn.RNN(input_size, hidden_size, num_layers)
output, hidden = rnn(input_data)` 

其中，`output`是RNN每个时间步的输出，`hidden`是最后一个时间步的隐藏状态。

# nn.LSTM
`nn.LSTM`是PyTorch中的一个循环神经网络模块，它基于长短期记忆（Long Short-Term Memory，LSTM）的架构。LSTM是一种特殊类型的循环神经网络，通过使用门控机制来解决传统循环神经网络中的梯度消失和梯度爆炸问题，从而能够更好地处理长期依赖关系。

`nn.LSTM`的主要参数包括：

*   `input_size`：输入数据的特征维度。
*   `hidden_size`：隐藏层的维度，也是LSTM单元输出的维度。
*   `num_layers`：LSTM的层数，默认为1。
*   `bias`：是否使用偏置，默认为True。
*   `batch_first`：输入数据的维度顺序是否为(batch, seq, feature)，默认为False。
*   `dropout`：是否应用dropout，用于防止过拟合，默认为0，表示不使用dropout。
*   `bidirectional`：是否使用双向LSTM，默认为False。

`nn.LSTM`的输入数据格式通常是一个三维张量，具体格式取决于`batch_first`参数的设置。如果`batch_first`为False（默认值），输入数据的维度应为(seq\_len, batch, input\_size)，其中seq\_len表示序列的长度，batch表示批次的大小，input\_size表示输入数据的特征维度。如果`batch_first`为True，输入数据的维度应为(batch, seq\_len, input\_size)。

`nn.LSTM`的前向传播过程会根据输入数据的时间步长和层数进行迭代计算，并返回最后一个时间步的输出以及最后一个时间步的隐藏状态和记忆细胞状态。这些输出可以用于下游任务，如分类或回归。

使用`nn.LSTM`时，可以通过调整参数来适应不同的任务和数据。此外，还可以使用`nn.LSTMCell`来构建自定义的LSTM网络。

`nn.LSTM`的返回值是一个元组，包含两个元素：output和(hidden\_state, cell\_state)。

1.  output：表示LSTM模型的隐藏状态输出。它是一个元组，包含了模型在每个时间步的输出结果。具体来说，output的形状是`(seq_len, batch, num_directions * hidden_size)`，其中：
    
    *   `seq_len`表示输入序列的长度；
    *   `batch`表示输入数据的批次大小；
    *   `num_directions`表示LSTM模型的方向数，通常为1或2（双向LSTM）；
    *   `hidden_size`表示隐藏状态的维度。
2.  (hidden\_state, cell\_state)：表示LSTM模型的最后一个时间步的隐藏状态和细胞状态。它们的形状都是`(num_layers * num_directions, batch, hidden_size)`，其中：
    
    *   `num_layers`表示LSTM模型的层数；
    *   `num_directions`表示LSTM模型的方向数，通常为1或2（双向LSTM）；
    *   `batch`表示输入数据的批次大小；
    *   `hidden_size`表示隐藏状态的维度。

这两个返回值可以用于进一步的处理和分析，比如用于序列标注、语言建模等任务。
也就是hidden\_state是output最后一个值，每个时间步都有一个cell\_state