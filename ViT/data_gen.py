import os
from PIL import Image
import natsort
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as tfs
import numpy as np
import torch.utils.data as data

GPU_NUMBER = 4


def load_class():
    return ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]


class FishDataSet(data.Dataset):
    def __init__(self, main_dir, transform):
        super(FishDataSet, self).__init__()
        self.main_dir = main_dir
        self.transform = transform
        self.data = []
        self.cls_dictionary = {
            "ALB": 0, "BET": 1, "DOL": 2, "LAG": 3,
            "NoF": 4, "OTHER": 5, "SHARK": 6, "YFT": 7}
        class_list = load_class()
        for cls_name in class_list:
            class_img = os.listdir(os.path.join(main_dir, cls_name))
            for img_name in class_img:
                des = os.path.join(main_dir, cls_name, img_name)
                self.data.append((des, self.cls_dictionary[cls_name]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_loc = self.data[idx][0]
        image = Image.open(img_loc).convert("RGB")
        image = self.transform(image)
        # tensor_image = tensor_image.unsqueeze(0)
        return image,  self.data[idx][1]
