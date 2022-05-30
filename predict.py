import cv2
import numpy as np
from crnn_model import crnn
import torch

char2num_dict = {'0': 0, '1': 1, '2': 2, '3': 3,
                 '4': 4, '5': 5, '6': 6, '7': 7,
                 '8': 8, '9': 9, '_': 10}
num2char_dict = {value: key for key, value in char2num_dict.items()}

model_dir = "model/model.pth"


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=10):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        # zheli
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [int(i) for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined


def single_recognition(img, model_dir):
    '''
    输入的是np array
    '''
    # 对读入图片做一些处理
    img = cv2.resize(img, (256, 32))
    img = img / 255.0 * 2.0 - 1.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = torch.tensor(img, dtype=torch.float)

    # 载入模型

    model_for_predict = crnn().float()
    model_for_predict.load_state_dict(torch.load(model_dir))
    model_for_predict.eval()

    y_pred_probMatrix = model_for_predict(img).squeeze()

    # Decode 阶段
    decoder = GreedyCTCDecoder(labels=num2char_dict)  # 使用的是最简单的贪婪算法
    y_pred_labels = decoder(y_pred_probMatrix)
    print(len(y_pred_labels))

    return y_pred_labels, y_pred_probMatrix


img = cv2.imread("test_images/7.jpeg", 0)
IMG = Image(img)
pred_labels, probMatrix = single_recognition(IMG.pos_img, model_dir)
print("识别结果为：", pred_labels, "长度为:", len(pred_labels))

# img = cv2.imread("test1.jpg", 0)
# pred_labels = single_recognition(img, model_dir)