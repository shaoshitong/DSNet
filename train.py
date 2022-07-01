import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import shutil
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from bisect import bisect_right
import time
import math
import dsnet
scaler=torch.cuda.amp.GradScaler()

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--data', default='/home/dataset/c100', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='dsnet50', type=str, help='network architecture')
parser.add_argument('--version', default='v1', type=str, help='network version')
parser.add_argument('--batch-size', type=int, default=12, help='batch size')

args = parser.parse_args()
num_classes = 100
trainset = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                [0.2675, 0.2565, 0.2761])
                                        ]))

testset = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                [0.2675, 0.2565, 0.2761]),
                                        ]))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=(torch.cuda.is_available()))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                    pin_memory=(torch.cuda.is_available()))

print('==> Building model..')
model = getattr(dsnet, args.arch)
net = model(expansion=4,num_classes=num_classes,version=args.version).cuda()
cudnn.benchmark = True


def train(epoch,criterion_cls,optimizer):
    total_loss=0.
    total_acc=0.
    net.train()
    for batch_idx, (input, target) in enumerate(trainloader):
        input = input.float().cuda()
        target = target.cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            logits = net(input)
            loss_cls = criterion_cls(logits, target)
        loss = loss_cls
        total_loss=total_loss+loss.item()*target.shape[0]
        total_acc=total_acc+(torch.argmax(logits,1)==target).sum().item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    total_acc/=len(trainloader.dataset)
    total_loss/=len(trainloader.dataset)
    print(f"train epoch:{epoch}, total_loss:{round(total_loss,2)}, total_acc:{round(total_acc*100,2)}%")

@torch.no_grad()
def test(epoch,criterion_cls):
    total_loss=0.
    total_acc=0.
    net.eval()
    for batch_idx, (inputs, target) in enumerate(testloader):
        inputs, target = inputs.cuda(), target.cuda()
        logits = net(inputs)
        loss_cls = criterion_cls(logits, target)
        total_loss = total_loss + loss_cls.item() * target.shape[0]
        total_acc = total_acc + (torch.argmax(logits, 1) == target).sum().item()
    total_acc/=len(testloader.dataset)
    total_loss/=len(testloader.dataset)
    print(f"test epoch:{epoch}, total_loss:{round(total_loss,2)}, total_acc:{round(total_acc*100,2)}%")
    return total_acc

if __name__ == '__main__':
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    epochs=200
    criterion_cls = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler=optim.lr_scheduler.MultiStepLR(optimizer,[100,150],gamma=0.1)

    for epoch in range(start_epoch, epochs):
        train(epoch, criterion_cls, optimizer)
        test_acc=test(epoch,criterion_cls)
        scheduler.step()
        if test_acc>best_acc:
            best_acc=test_acc
    print(f"best acc: {round(best_acc*100,2)}%")


