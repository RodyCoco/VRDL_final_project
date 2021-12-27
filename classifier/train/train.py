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
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import antialiased_cnns
import os


def default_loader(path):
    return Image.open(path).convert('RGB')


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

class MyDataset(Dataset):
    def __init__(self, transform=None, loader=default_loader, image_folder = 'crop_train/'):
        imgs = []
        file_name = ['ALB','BET','DOL','LAG','NoF','OTHER','SHARK','YFT']
        for i in range(len(file_name)):
            all_file = os.listdir(image_folder + file_name[i])
            for file in all_file:
                imgs.append((image_folder + file_name[i] + '/' + file,i))



        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        
        if self.transform is not None:
            # print(fn)
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # data, target = torch.Tensor(data).to(device), torch.Tensor(target).to(device)
        optimizer.zero_grad()
        output = model(data)



        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(args, model, device, test_loader):
    model.eval()
    with torch.no_grad():
        all = 0
        correct = 0
        for data, target in test_loader:
            answer = target
            data, target = data.to(device), target.to(device)
            # data, target = torch.Tensor(data).to(device), torch.Tensor(target).to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            # get the index of the max log-probability
            # print(pred.cpu()[0][0])
            # print(answer[0])
            if pred.cpu()[0][0] == answer[0]:
                correct +=1
            all += 1
    print('accuracy: ', correct/all)




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=25, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--patience', type=int, default=10, metavar='N',
                        help='input patience for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.5, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    
    device = torch.device("cuda")

    
    
    

    # transform2 = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    #     ]
    # )
    
    
    


        

    # model = antialiased_cnns.resnext101_32x8d(pretrained=True)
    # model.fc = nn.Linear(2048, 200) 
    model = models.resnext101_32x8d(pretrained=True)
    
    model.fc = nn.Linear(2048, 8) 
    
    child_counter = 0
    for child in model.children():
        if child_counter < 7:
            print("child  ",child_counter," was frozen")
            for param in child.parameters():
                param.requires_grad = False
        elif child_counter == 6:
            children_of_child_counter = 0
            for children_of_child in child.children():
                if children_of_child_counter < 5:
                    for param in children_of_child.parameters():
                        param.requires_grad = False
                    print('child ', children_of_child_counter, 'of child',child_counter,' was frozen')
                else:
                    print('child ', children_of_child_counter, 'of child',child_counter,' was not frozen')
                children_of_child_counter += 1

        else:
            print("child ",child_counter," was not frozen")
        child_counter += 1
    
    model.to(device)
    transform_blur_set = [
        transforms.GaussianBlur(9, sigma=(1, 7)),
        transforms.GaussianBlur(7, sigma=(1, 7)),
        transforms.GaussianBlur(5, sigma=(1, 7)),
        transforms.GaussianBlur(3, sigma=(1, 7))
    ]
    transform_blur = [transforms.RandomChoice(transform_blur_set)]
    
    transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(
                brightness=(0.7, 1.4),
                contrast=(0.7, 1.4),
                saturation=(0.7, 1.4),
            ),
            transforms.RandomApply(transform_blur, p=0.75),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((0,90), expand=True),
            transforms.RandomAffine(degrees=(0, 0),shear=(20)),
            transforms.Resize((512,512)),
            # transforms.Pad((256,256,256,256),padding_mode = 'symmetric'),
            # transforms.Pad((256,256,256,256),padding_mode = 'edge'),
            # transforms.RandomAffine(degrees=(-60, 60), translate=(0.15, 0.15), scale=(0.3, 1.8), shear=(0.2)),
            # transforms.CenterCrop(300),
            # transforms.RandomErasing(p = 1,ratio =(1,1), scale=(0.05,0.15)),
            
        ]
    )
    transform_ori = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.Resize((224,224)),
        # transforms.ToPILImage()

        
        ]
    )
    


    train_data=MyDataset(transform=transform2)
    train_loader = DataLoader(
        train_data,
        args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        # collate_fn=my_collate
    )

    test_data = MyDataset(transform=transform_ori,image_folder='crop_validate/')
    test_loader = DataLoader(
        test_data,
        1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        # collate_fn=my_collate
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, cycle_momentum=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)

    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # optimizer = optim.Adam(model.parameters(), lr=1e-5)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-6, max_lr=5e-2, cycle_momentum=False)

        
    
    
    for epoch in range(1,args.epochs+1):
    
        train(args, model, device, train_loader, optimizer, epoch)
        

        if epoch % 1 == 0:
            test(args, model, device, train_loader)
            torch.save(model, 'resnext101_'+str(epoch) + "model.pt")
        
        scheduler.step()

    
    
    


if __name__ == '__main__':    
    main()
    