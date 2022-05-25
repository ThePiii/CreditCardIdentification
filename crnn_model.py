import torch
import torch.nn as nn

# 怎么读数据

# 这个类用来实现keras中Bidirectional里面merge_mode=sum的参数
class BiLSTM_add(nn.Module):
    def __init__(self):
        super(BiLSTM_add, self).__init__()


    def forward(self,input):
        _,_,n = input.shape
        input[:,:,:int(n/2)] = input[:,:,:int(n/2)] + input[:,:,int(n/2):]
        return input[:,:,:int(n/2)]

class Permute(nn.Module):
    def __init__(self, orders):
        super(Permute, self).__init__()
        self.order = orders

    def forward(self, x):
        # 如果数据集最后一个batch样本数量小于定义的batch_batch大小，会出现mismatch问题。可以自己修改下，如只传入后面的shape，然后通过x.szie(0)，来输入。
        return torch.permute(x,self.order)


class SelectItem(nn.Module):
    def __init__(self,item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self,inputs):
        return inputs[self.item_index]

    # img_size = (height:256, width:32)

class crnn(nn.Module):
    def __init__(self):
        super(crnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3,stride = 1,padding='same',bias=True),   # input (N,C,H,W)  (64,1,32,256)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, 3,stride = 1,padding='same',bias=True),  # (64,64,16,128)
            nn.BatchNorm2d(128),
            nn.ReLU(),  # kernel_size, stride
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, 3,stride = 1,padding='same',bias=True),  # (64,128,8,64)
            nn.BatchNorm2d(256),
            nn.ReLU(),  # kernel_size, stride
            nn.Conv2d(256, 256, 3,stride = 1,padding='same',bias=True),  # (64,128,8,64)
            nn.BatchNorm2d(256),
            nn.ReLU(),  # kernel_size, stride
            nn.MaxPool2d((2, 1),stride=(2,1)),

            nn.Conv2d(256, 512, 3, stride=1, padding='same', bias=True),  # (64,256,4,64)
            nn.BatchNorm2d(512),
            nn.ReLU(),  # kernel_size, stride
            nn.Conv2d(512, 512, 3, stride=1, padding='same', bias=True),  # (64,512,4,64)
            nn.BatchNorm2d(512),
            nn.ReLU(),  # kernel_size, stride
            nn.MaxPool2d((2, 1), stride=(2, 1)),

            nn.Conv2d(512, 512, 3, stride=1, padding='same', bias=True),  # (64,512,2,64)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )  # (64,512,1,64)

        self.rnn = nn.Sequential(
            nn.LSTM(bidirectional=True, hidden_size=256, input_size=64,batch_first=True),  #   input_size (N,L,H) -> (64,512,64)
            SelectItem(0),
        # keras 里面的bilstm可以指定合并方式，但是pytorch里不能,但有必要搞么
            BiLSTM_add(), # (64, 512, 512)_
            nn.LayerNorm(256),  # (64,512,256)  input_size(N,L,H) -> (64,512,256)
            nn.LSTM(bidirectional=True, hidden_size=256, input_size=256,batch_first=True),  # (64,512,256)
            SelectItem(0),
            nn.Linear(in_features=512, out_features=11),  # (64,512,11)
            # 这里和原代码不一样，原代码好像是直接用softmax
            nn.LogSoftmax(dim=-1) # ()
        )

    def forward(self, img):
        feature = self.conv(img)
        feature = feature.squeeze()
        output = self.rnn(feature)
        return output
