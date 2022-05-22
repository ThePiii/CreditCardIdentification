# encoding: utf-8
import os
import re
import pandas as pd

def to_text(src, dst1):
    txt = []
    filenames = os.listdir(src)
    # filenames.sort(key= lambda x:(int((x.split('_')[1]).split('.')[0]))) # train_2045.png
    # filenames.sort(key =lambda x: int(x.split('.')[0]))  # 2045.png
    # filenames.sort(key=lambda x: re.search("^[0-9]*", x.split('.')[0]).group()[0:-1])
    labels = []

    for filename in filenames:
        label = re.search("^[0-9_]*", filename.split('.')[0]).group()
        labels.append(label)
        print(filename, label)

    df = pd.DataFrame([filenames, labels])


    # print(filenames)

    # for item in filenames:
    #     if item.endswith('.png'):
    #         txt.append(item)

    fo = open(dst1, 'w')

    for i in range(len(filenames)):
        fo.write(filenames[i] + " " +  labels[i] + "\n")

    fo.close()


if __name__ == "__main__":
    src = "train_img"
    dst1 = "train.txt"
    to_text(src, dst1)
