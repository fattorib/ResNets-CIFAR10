import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Model import ResNet, ResNetV2
from Train_Model import Train_Model
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import time


# Data augmentation as outlined in paper
transform_train = transforms.Compose([
    transforms.RandomCrop((32, 32), padding=4, fill=0,
                          padding_mode='constant'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


train_data = CIFAR10(root='/CIFAR', train=True,
                     download=False, transform=transform_train)

test_data = CIFAR10(root='/CIFAR', train=False,
                    download=False, transform=transform_test)

# Getting a validation set
train_data, val_data = torch.utils.data.random_split(train_data, [
    45000, 5000])


# ResNet20
# Paper acc: 91.25%
print('-----ResNet20-----')
model = ResNet(filters_list=[16, 32, 64], N=3)
model.load_state_dict(torch.load('ResNets/models/resnet20.pth'))
model.cuda()
model_class = Train_Model(model, train_data, test_data, val_data)
model_class.eval(train=False)


# ResNet32
# Paper acc: 92.49%
print('-----ResNet32-----')
model = ResNet(filters_list=[16, 32, 64], N=5)
model.load_state_dict(torch.load('ResNets/models/resnet32.pth'))
model.cuda()
model_class = Train_Model(model, train_data, test_data, val_data)
model_class.eval(train=False)


# ResNet44
# Paper acc: 92.83
print('-----ResNet44-----')
model = ResNet(filters_list=[16, 32, 64], N=7)
model.load_state_dict(torch.load('ResNets/models/resnet44.pth'))
model.cuda()
model_class = Train_Model(model, train_data, test_data, val_data)
model_class.eval(train=False)

# ResNet56
# Paper acc: 93.03
print('-----ResNet56-----')
model = ResNet(filters_list=[16, 32, 64], N=9)
model.load_state_dict(torch.load('ResNets/models/resnet56.pth'))
model.cuda()
model_class = Train_Model(model, train_data, test_data, val_data)
model_class.eval(train=False)
