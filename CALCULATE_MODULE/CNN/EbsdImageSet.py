import itertools
import os
import random
from random import sample

import cv2
import torch
from PIL import Image
from torch import tensor
from torch.utils.data import Dataset

from CALCULATE_MODULE.GNN.data_helper import Sample
from utils import data_util


class EBSDImageSet(Dataset):#需要继承torch.utils.data.Dataset
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.images[index], self.labels[index]
        if isinstance(img_path, list):
           image_list = []
           for path_item in img_path:
               image = Image.open(path_item)
               image = image.convert("RGB")
               if self.transform:
                   image = self.transform(image)
               image_list.append(image)
           return torch.stack(image_list), tensor(label)
        else:
            image = Image.open(img_path)
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, tensor(label)

    def __len__(self):
        # 返回数据集的总大小。
        return len(self.images)



def get_ebsd_image_sets(test_size=0.2, target="block",
                        label_options=["ys","uts","el"], directions=["90"],
                        dir_path="./image_data/subgraph4", random_state=None):
    random.seed(random_state)
    images = {"train":[], "test":[]}
    labels = {"train":[], "test":[]}
    if target == "sub":
        for sample_id in os.listdir(dir_path):
            for block_id in os.listdir(os.path.join(dir_path, sample_id)):
                label = [Sample(sample_id).prop(l, d) for l, d in itertools.product(label_options, directions)]
                sub_graphs = os.listdir(os.path.join(dir_path, sample_id, block_id))
                sub_graphs = [os.path.join(os.path.dirname(__file__),dir_path,sample_id,block_id,g) for g in sub_graphs]
                test_graphs, train_graphs = data_util.split(sub_graphs, test_size)
                images["train"].extend(train_graphs)
                labels["train"].extend([label] * len(train_graphs))
                images["test"].extend(test_graphs)
                labels["test"].extend([label] * len(test_graphs))
    elif target == "block":
        for sample_id in os.listdir(dir_path):
            block_dirs = os.listdir(os.path.join(dir_path, sample_id))
            block_num = len(block_dirs)
            num_of_testset = test_size if isinstance(test_size, int) else int(block_num * test_size)  # 每个样品的测试block数量
            test_block_ids = [random.randint(1, block_num) for j in  # 测试block的id ，如果test_size<＜0, 那test_block_ids就为空
                              range(num_of_testset)]
            for block_id in block_dirs:
                type = "test" if int(block_id) in test_block_ids else "train"
                label = [Sample(sample_id).prop(l, d) for l, d in itertools.product(label_options, directions)]
                sub_graphs = os.listdir(os.path.join(dir_path, sample_id, block_id))
                sub_graphs = [os.path.join(os.path.dirname(__file__),dir_path,sample_id,block_id,g) for g in sub_graphs]
                images[type].extend(sub_graphs)
                labels[type].extend([label] * len(sub_graphs))
    elif target == "sample":
        pass
    else:
        raise Exception("不支持此类目标任务！")

    return images,labels


if __name__ == '__main__':
    pass