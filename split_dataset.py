import os
import random
import shutil

random.seed(0)

split_name = 'val'
split_pct = 0.2  # split out 1% for validation

data_dir = os.path.join(os.curdir, 'train')
for root, dirs, files in os.walk(data_dir):
    for sub_dir in dirs:  # merely use the first iter
        # get a list of image names and shuffle them
        images = os.listdir(os.path.join(root, sub_dir))
        images = list(filter(lambda x: x.endswith('.jpg'), images))
        num_images = len(images)
        random.shuffle(images)

        num_split = int(num_images * split_pct)  # how many images will be split out

        for i in range(num_split):
            out_dir = os.path.join(os.curdir, split_name, sub_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            to_path = os.path.join(out_dir, images[i])
            from_path = os.path.join(data_dir, sub_dir, images[i])
            shutil.move(from_path, to_path)

