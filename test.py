from AlexNet import alexnet
from torchvision import datasets, transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

data_root = os.path.abspath(os.path.join(os.getcwd(), "data"))
pretrained_path = os.path.abspath(os.path.join(os.getcwd(), "checkpoints/AlexNet202011260351.pth"))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

if __name__ == '__main__':
    train_set = datasets.CIFAR100(root=data_root, train=True, download=True, transform=data_transform['train'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=1)
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        for label in labels:
            print(label)
        time.sleep(0.5)

    models.alexnet()

