import torch
import numpy as np
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from PIL import Image
from model import Vit_large_patch16_224, Vit_large_patch16_384
from data_gen import load_class, GPU_NUMBER, FishDataSet
import csv
import os
import json

batch_size = 1
num_model = 5
test_time = 1

GPU_NUMBER = 8
test_trans = tfs.Compose([
    tfs.Resize((384, 384), Image.BILINEAR),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
])


def get_result(root, model, img_name, is_croped, bbox):
    result = 0
    for i in range(test_time):
        des = os.path.join(root, img_name)
        image = Image.open(des).convert("RGB")
        if is_croped:
            image = image.crop(bbox)
        image = test_trans(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            out = \
                model(image.type(torch.DoubleTensor).cuda(GPU_NUMBER)).double()
            out = torch.nn.Softmax(dim=1)(out)
            out = out.squeeze(0).cpu().detach().numpy()
            result += out
    result = result/test_time

    return result


def procedure():
    fish_class = load_class()
    model = Vit_large_patch16_384().cuda(GPU_NUMBER)
    model = torch.nn.DataParallel(model, device_ids=[8, 4, 9, 5])
    name = f"Vit_large_patch16_384_bbox_1.pkl"
    print(name)
    model.load_state_dict(torch.load(name))
    model.double()
    model.eval()

    print("load model done")
    bbox_dir = "../faster_rcnn/annotations_sep_detector_2classes_e15"

    with open(f'{name[:-4]}_bbox.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image"] + fish_class)

        stg1_root = "../test_stg1"
        stg1_dir = os.listdir(stg1_root)
        stg1_dir.sort()
        get_pred(bbox_dir, stg1_root, writer, model, False)

        stg1_root = "../test_stg2"
        stg1_dir = os.listdir(stg1_root)
        stg1_dir.sort()
        get_pred(bbox_dir, stg1_root, writer, model, True)


def get_pred(bbox_dir, root, writer, model, is_stg2):
        stg1_dir = os.listdir(root)
        stg1_dir.sort()
        for img_name in stg1_dir:
            with open(os.path.join(bbox_dir, img_name[:-3]+"json")) as f:
                data = json.load(f)
                bbox = data["annotations"][0]["bbox"]
                if bbox != []:
                    left, up, width, height = bbox
                    bbox = [left, up, left+width, up+height]
                try:
                    score = data["annotations"][0]["score"]
                except:
                    score = 0

            if score >= 0.3:
                result = get_result(root, model, img_name, True, bbox)
            else:
                result = get_result(root, model, img_name, False, bbox)

            if is_stg2:
                writer.writerow(["test_stg2/"+img_name] + list(result))
            else:
                writer.writerow([img_name] + list(result))

if __name__ == '__main__':
    procedure()
