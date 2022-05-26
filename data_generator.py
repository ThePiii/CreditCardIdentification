
import torch
import numpy as np
import cv2

# 只有数字的版本
char2num_dict = {'0': 0, '1': 1, '2': 2, '3': 3,
                 '4': 4, '5': 5, '6': 6, '7': 7,
                 '8': 8, '9': 9, '_': 10}
num2char_dict = {value: key for key, value in char2num_dict.items()}


class DataGenerator:
    '''
    这个类是从txt文件里面读取图片文件名，以及对应的gt
    '''

# down_sample_factor 是 4 是因为 每张图片里有4个数字
    def __init__(self, txt_file_path, img_size, down_sample_factor, batch_size, max_label_length):
        self.txt_file_path = txt_file_path
        self.img_w, self.img_h = img_size
        self.batch_size = batch_size
        self.max_label_length = max_label_length
        # 这是每个数字对应的列数
        self.each_pred_label_length = int(self.img_w // down_sample_factor)

        # 从train.txt文件中获取文件名以及所对应的标签,其中每一行形式为 train_img/0000a_0.png 0000
        data_txt = open(self.txt_file_path, "r")
        data_txt_list = data_txt.readlines()
        # train_img/0000a_0.png
        self.img_list = [line.split(" ")[0]
                         for line in data_txt_list]  # 里面保存了所有的数据的文件名
        self.img_list = np.array(self.img_list)  # ] for l 将他转换成array
        # 0000
        self.img_labels_chars_list = [line.split("\n")[0].split(" ")[
            1:] for line in data_txt_list]
        self.img_labels_chars_list = np.array(self.img_labels_chars_list,dtype=object)
        #多少张图片
        self.img_number = len(self.img_list)
        data_txt.close()

        index = np.random.permutation(self.img_list.shape[0])
        self.img_list = self.img_list[index]
        self.img_labels_chars_list = self.img_labels_chars_list[index]
        self.char2num_dict = char2num_dict
        self.num2char_dict = num2char_dict

    def get_data(self, is_training=True):

        labels_length = np.zeros((self.batch_size, 1))
        # 就是说每个数字占了几列像素
        pred_labels_length = np.full(
            (self.batch_size, 1), self.each_pred_label_length, dtype=np.float64)
        while True:
            data, labels = [], []
            #随机不重复抽取batch_size张图片的index
            to_network_idx = np.random.choice(
                self.img_number, self.batch_size, replace=False)

            #对应图片文件名
            img_to_network = self.img_list[to_network_idx]
            #对应label
            correspond_labels = self.img_labels_chars_list[to_network_idx]
            for i, img_file in enumerate(img_to_network):
                img = cv2.imread(img_file)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_img = cv2.resize(gray_img, (self.img_w, self.img_h))
                gray_img = gray_img.astype(np.float32)

                data.append(gray_img)
                str_label = correspond_labels[i]
                labels_length[i][0] = len(str_label[0])
                #他这里把label里面的_识别成10了，这样可以么？
                num_label = [char2num_dict[ch] for ch in str_label[0]]

                for n in range(self.max_label_length - len(str_label[0])):
                    num_label.append(self.char2num_dict['_'])

                labels.append(num_label)
            data = np.array(data, dtype=np.float64) / 255.0 * 2 - 1  # 零中心化
            data = np.expand_dims(data, axis=1)  # 这里改了一下让维度对应后面net的输入
            labels = np.array(labels, dtype=np.float64)
            inputs = {"targets": torch.tensor(labels, dtype=torch.float),   # (64,26)  图片的真实标签，为什么是26
                      "pic_inputs": torch.tensor(data, dtype=torch.float),
                      "input_lengths": torch.tensor(pred_labels_length, dtype=torch.int),    # (64,1) 值为64,指输入的个数
                      "target_lengths": torch.tensor(labels_length, dtype=torch.int)}  # (64,1) 值为4，指label里有几个字符
            # outputs 存在意义？
            outputs = {"ctc_loss_output": torch.zeros(
                (self.batch_size, 1), dtype=torch.float)}
            if is_training:
                yield (inputs, outputs)
            else:
                yield (data, pred_labels_length)

# 各种路径 以及参数
# weight_save_path = "model/"
# 数字训练路径





