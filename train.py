import torch
import torch.nn as nn
from data_generator import DataGenerator
from crnn_model import crnn
from pathlib import Path

train_txt = "train.txt"
val_txt_path = "val.txt"
img_size = (256, 32) # W*H
# 各种训练时候的参数
num_classes = 11 # 包含“blank”
max_label_length = 26  # 为什么是26
down_sample_factor = 4
epochs = 100
batch_size = 64

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

save_PATH = "model/model.pth"
save_PATH_drive = Path("/content/drive/MyDrive/model/model.pth")

def train():
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.kaiming_normal_(m.bias)
        elif isinstance(m, nn.LSTM):
            nn.init.kaiming_normal_(m.weight)
            nn.init.kaiming_normal_(m.bias)
    train_data = DataGenerator(train_txt, img_size, down_sample_factor, batch_size, max_label_length)
    loss = nn.CTCLoss(blank=10)
    loss = loss.to(device)
    net = crnn().float()
    net = net.to(device)

    try:
        net.load_state_dict(torch.load(save_PATH_drive))
        print("restore model successful...")
    except:
        print("Create new model...")
        pass

    # train_ls = []
    optimizer = torch.optim.Adam(params=net.parameters())
    for epoch in range(epochs):
        train_ls_temp = 0
        count = 0
        for X, _ in train_data.get_data():
            img = X['pic_inputs']
            img = img.to(device)
            input_lengths = X['input_lengths']
            targets = X['targets']   # 好像是输入图片label数
            targets = targets.to(device)
            target_lengths = X['target_lengths']
            log_probs = net(img)
            net = net.float()
            optimizer.zero_grad() # 一定要清零
            l = loss(log_probs, targets, input_lengths, target_lengths) # log_probs,targets,input_lengths,target_lengths
            l.backward()
            optimizer.step()
            train_ls_temp += l
            count += 1
            # print(train_ls_temp)
            if count % 100 == 0:
                print("---- 训练100个数据了, 保存一下模型 ----")
                torch.save(net.state_dict(), save_PATH)
                if save_PATH_drive.is_file() and count % 500 == 0:
                    print("---- 训练500个数据了，保存一下模型到Drive ----")
                    torch.save(net.state_dict(), save_PATH_drive)
        train_ls.append(train_ls_temp)
        print("第{}轮训练的loss：{}".format(epoch, train_ls_temp))
        torch.save(net.state_dict(), save_PATH)
        if save_PATH_drive.is_file():
            torch.save(net.state_dict(), str(save_PATH_drive))

    return train_ls



if __name__ == '__main__':
    train()
    plt.plot(train_ls, '-')
    plt.show()