import torch
import numpy as np
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from PIL import Image
from model import Vit_large_patch16_384, Vit_large_patch16_224
from data_gen import load_class, GPU_NUMBER, FishDataSet
import csv
import os

batch_size = 1
num_model = 5
test_time = 1

GPU_NUMBER = 5
test_trans = tfs.Compose([
    tfs.Resize((384, 384), Image.BILINEAR),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
])

def procedure():
    fish_class = load_class()
    model = Vit_large_patch16_384().cuda(GPU_NUMBER)
    model = torch.nn.DataParallel(model, device_ids=[5,4,8,9])
    name = f"Vit_large_patch16_384_1.pkl"
    print(name)
    model.load_state_dict(torch.load(name))
    model.double()
    model.eval()

    print("load model done")

    with open(f'{name[:-4]}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image"] + fish_class)

        stg1_root = "../test_stg1"
        stg1_dir = os.listdir(stg1_root)
        stg1_dir.sort()
        for img_name in stg1_dir:
            sum = 0
            for i in range(test_time):
                des = os.path.join(stg1_root, img_name)
                image = Image.open(des).convert("RGB")
                image = test_trans(image)
                image = image.unsqueeze(0)
                with torch.no_grad():
                    out = model(image.type(torch.DoubleTensor).cuda(GPU_NUMBER)).double()
                    out = torch.nn.Softmax(dim=1)(out)
                    out = out.squeeze(0).cpu().detach().numpy()
                    sum += out
            sum = sum/test_time
            writer.writerow([img_name] + list(sum))

        stg2_root = "../test_stg2"
        stg2_dir = os.listdir(stg2_root)
        stg2_dir.sort()
        for img_name in stg2_dir:
            sum = 0
            for i in range(test_time):
                des = os.path.join(stg2_root, img_name)
                image = Image.open(des).convert("RGB")
                image = test_trans(image)
                image = image.unsqueeze(0)
                with torch.no_grad():
                    out = model(image.type(torch.DoubleTensor).cuda(GPU_NUMBER)).double()
                    out = torch.nn.Softmax(dim=1)(out)
                    out = out.squeeze(0).cpu().detach().numpy()
                    sum += out
            sum = sum/test_time
            writer.writerow(["test_stg2/"+img_name] + list(sum))

if __name__ == '__main__':
    procedure()
