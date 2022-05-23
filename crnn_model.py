import torch
import torch.nn as nn

# 怎么读数据

# 这个类用来实现keras中Bidirectional里面merge_mode=sum的参数
class BiLSTM_add(nn.Module):
    def __init__(self):
        super(BiLSTM_add, self).__init__()


    def forward(self,input):
        _,n = input.shape
        input[:,:int(n/2)] = input[:,:int(n/2)] + input[:,int(n/2):]
        return input[:,:int(n/2)]

class SelectItem(nn.Module):
    def __init__(self,item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self,inputs):
        return inputs[self.item_index]

    # img_size = (height:256, width:32)
    input = torch.tensor().view(picture_height, picture_width)

def net():

# 记得返回来写一下权重初始化
    cnn_part = nn.Sequential(
        nn.Conv2d(1, 64, 3,stride = 1,padding='same',bias=True),
        nn.BatchNorm1d(),
        nn.ReLU(),  # kernel_size, stride
        nn.MaxPool2d((2, 2)),

        nn.Conv2d(64, 128, 3,stride = 1,padding='same',bias=True),  # in_channels, out_channels, kernel_size
        nn.BatchNorm1d(),
        nn.ReLU(),  # kernel_size, stride
        nn.MaxPool2d((2, 2)),

        nn.Conv2d(128, 256, 3,stride = 1,padding='same',bias=True),  # in_channels, out_channels, kernel_size
        nn.BatchNorm1d(),
        nn.ReLU(),  # kernel_size, stride
        nn.Conv2d(256, 256, 3,stride = 1,padding='same',bias=True),  # in_channels, out_channels, kernel_size
        nn.BatchNorm1d(),
        nn.ReLU(),  # kernel_size, stride
        nn.MaxPool2d((2, 1),stride=(2,1)),

        nn.Conv2d(256, 512, 3, stride=1, padding='same', bias=True),  # in_channels, out_channels, kernel_size
        nn.BatchNorm1d(),
        nn.ReLU(),  # kernel_size, stride
        nn.Conv2d(512, 512, 3, stride=1, padding='same', bias=True),  # in_channels, out_channels, kernel_size
        nn.BatchNorm1d(),
        nn.ReLU(),  # kernel_size, stride
        nn.MaxPool2d((2, 1),stride=(2,1)),

        nn.Conv2d(512, 512, 3, stride=1, padding='same', bias=True),  # in_channels, out_channels, kernel_size
        nn.BatchNorm1d(),
        nn.ReLU(),
        nn.MaxPool2d((2, 1)),


    # rnn_part
        nn.LSTM(bidirectional=True, hidden_size=256, input_size=512),
        SelectItem(0),
    # keras 里面的bilstm可以指定合并方式，但是pytorch里不能,但有必要搞么
        BiLSTM_add(),
        nn.BatchNorm1d(),
        nn.LSTM(bidirectional=True, hidden_size=256, input_size=256),
        SelectItem(0),
        nn.Linear(in_features=512, out_features=11),
        nn.Softmax()
    )

y_pred = net(input)
loss = nn.CTCLoss(blank=)
loss = loss(input,target, input_lengths, target_lengths)
ctcloss()

