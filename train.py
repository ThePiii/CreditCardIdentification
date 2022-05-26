import torch
import torch.nn as nn
from data_generator import DataGenerator
from crnn_model import crnn

train_txt = "train.txt"
val_txt_path = "val.txt"
img_size = (256, 32) # W*H
# 各种训练时候的参数
num_classes = 11 # 包含“blank”
max_label_length = 26  # 为什么是26
down_sample_factor = 4
epochs = 100
batch_size = 64


save_PATH = "model/model.pt"

def train():
    train_data = DataGenerator(train_txt, img_size, down_sample_factor, batch_size, max_label_length)
    loss = nn.CTCLoss(blank=10)
    net = crnn().float()

    try:
        net.load_state_dict(torch.load(PATH))
        print("restore model successful...")
    except:
        print("Create new model...")
        pass

    train_ls = []
    optimizer = torch.optim.Adam(params=net.parameters())
    for epoch in range(epochs):
        train_ls_temp = 0
        for X, _ in train_data.get_data():
            img = X['pic_inputs']
            input_lengths = X['input_lengths']
            targets = X['targets']   # 好像是输入图片label数
            target_lengths = X['target_lengths']
            log_probs = net(img)
            net = net.float()
            optimizer.zero_grad()
            l = loss(log_probs, targets, input_lengths, target_lengths) # log_probs,targets,input_lengths,target_lengths
            l.backward()
            optimizer.step()
            train_ls_temp += l
        train_ls.append(train_ls_temp)
    print(train_ls[:10])



if __name__ == '__main__':
    train()