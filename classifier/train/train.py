#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from PIL import ImageOps
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import antialiased_cnns
import os
from tqdm import tqdm
from model import swin_large_patch4_window7_224
import warnings

warnings.filterwarnings('ignore')


def default_loader(path):
    return Image.open(path).convert('RGB')


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]


class MyDataset(Dataset):

    def __init__(self, transform=None, loader=default_loader):
        imgs = []
        file_name = [
            'ALB',
            'BET',
            'DOL',
            'LAG',
            'NoF',
            'OTHER',
            'SHARK',
            'YFT',
            ]
        for i in range(len(file_name)):
            all_file = os.listdir(file_name[i])
            for file in all_file:
                imgs.append((file_name[i] + '/' + file, i))

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        (fn, label) = self.imgs[index]
        img = self.loader(fn)
        (w, h) = img.size
        img = ImageOps.pad(img, (max(w, h), max(w, h)))

        if self.transform is not None:

            # print(fn)

            img = self.transform(img)

        return (img, label)

    def __len__(self):
        return len(self.imgs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(
        args,
        model,
        device,
        train_loader,
        optimizer, epoch):

    model.train()
    for (batch_idx, (data, target)) in enumerate(train_loader):
        (data, target) = (data.to(device), target.to(device))
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            lr = get_lr(optimizer)
            print('Train:{} [{:4d}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(),
                lr,
                ))
            if args.dry_run:
                break


def test(
        args,
        model,
        device,
        test_loader):
    model.eval()
    with torch.no_grad():
        all = 0
        correct = 0
        for (data, target) in tqdm(test_loader):
            answer = target
            (data, target) = (data.to(device), target.to(device))
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(len(data)):
                all += 1
                if pred[i][0] == target[i]:
                    correct += 1
    return correct / all


def main():

    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=25,
                        metavar='N',
                        help='input batch size for training (default: 100)'
                        )
    parser.add_argument('--patience', type=int, default=10, metavar='N',
                        help='input patience for training (default: 10)'
                        )
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)'
                        )
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)'
                        )
    parser.add_argument('--no-cuda', action='store_true',
                        default=False, help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true',
                        default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100,
                        metavar='N',
                        help='how many batches to wait'
                        )
    parser.add_argument('--save-model', action='store_true',
                        default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda')

    # transform2 = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    #     ]
    # )

    model = antialiased_cnns.resnext101_32x8d(pretrained=True)
    model.fc = nn.Linear(2048, 8)

    # freeze

    child_counter = 0
    for child in model.children():
        if child_counter < 6:
            print('child  ', child_counter, ' was frozen')
            for param in child.parameters():
                param.requires_grad = False
        elif child_counter == 6:
            children_of_child_counter = 0
            for children_of_child in child.children():
                if children_of_child_counter < 5:
                    for param in children_of_child.parameters():
                        param.requires_grad = False
                    print('child ', children_of_child_counter,
                          'of child', child_counter, ' was frozen')
                else:
                    print('child ', children_of_child_counter,
                          'of child', child_counter, ' was not frozen')
                children_of_child_counter += 1
        else:

            print('child ', child_counter, ' was not frozen')
        child_counter += 1

    model.to(device)
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229,
                             0.224, 0.225)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation((0, 90), expand=True),
        transforms.Resize((224, 224)),
        ])
    transform_ori = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                        mean=(0.485, 0.456, 0.406),
                                        std=(0.229, 0.224, 0.225)),
                                        transforms.Resize((224, 224))])

    train_data = MyDataset(transform=transform2)
    train_loader = DataLoader(train_data, args.batch_size,
                              shuffle=True, num_workers=8,
                              pin_memory=False)

    test_data = MyDataset(transform=transform_ori)
    test_loader = DataLoader(test_data, 8, shuffle=True, num_workers=1,
                             pin_memory=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-4,
                           weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=0,
        cooldown=1,
        min_lr=1e-8,
        verbose=True,
        )

    for epoch in range(1, args.epochs + 1):

        train(
            args,
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            )
        acc = test(args, model, device, test_loader)
        print(acc)
        torch.save(model, 'model.pt')
        scheduler.step(acc)


if __name__ == '__main__':
    main()
