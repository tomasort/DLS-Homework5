import torch
from torch import nn
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.optim as optim
from resnet import ResNet, BasicBlock, test
from trainer import train_model
import argparse
import os


parser = argparse.ArgumentParser(description='Perform transfer learning')
parser.add_argument('--gpu_name', type=str)
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. The default is 0.1')
parser.add_argument('--save', type=str, default=f"./problem5/batch_size_experiments/",
                    help='The directory to save the data from the model')
parser.add_argument('--batch', type=int, default=128,
                    help='The directory to save the data from the model')
parser.add_argument('--epochs', default=350, type=int,
                    help='Number of epochs')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--target-acc', default=0.92, type=float,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dataset', type=str, default='./data')

args = parser.parse_args()

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, 4),
     transforms.ToTensor(),
     normalize])

trainset = datasets.CIFAR10(root=args.dataset, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                          shuffle=True, num_workers=2)
testset = datasets.CIFAR10(root=args.dataset, train=False,
                                       download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                         shuffle=False, num_workers=2)

def resnet50():
    return ResNet(BasicBlock, [8, 8, 8])

net = resnet50()
test(net)
save_path = os.path.join(args.save, args.gpu_name, f"resnet_{int(time.time())%100_000}")
dataloaders = {"train": trainloader, "val": testloader}
model = resnet50()
params_to_update = model.parameters()
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200, 300])
train_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs, scheduler=scheduler, save_path=save_path, model_name=f"resnet50_{args.batch}", target_accuracy=0.92)