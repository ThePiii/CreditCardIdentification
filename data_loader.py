from torch.utils.data import Dataset
import cv2
import os

txt = open("train.txt")
l = txt.readlines()
path, labels = [], []

for line in l:
    temp = line[0:-1].split(" ")
    path.append(temp[0])
    labels.append(temp[1])


class MyData(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.img_path = path

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img = cv2.imread(img_name)
        label = labels[idx]
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "train_img"

dataset = MyData(root_dir)
img, label = dataset.__getitem__(1)

print(label)