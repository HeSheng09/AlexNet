from AlexNet import alexnet
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import argparse


def parse_args():
    """
      Parse input arguments
      """
    parser = argparse.ArgumentParser(description='Train a AlexNet network')
    parser.add_argument('--num_classes', dest='num_classes',
                        help='num_classes',
                        default=10, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs',
                        help='num_epochs',
                        default=2, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        help='batch_size',
                        default=8, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                        help='num_workers',
                        default=2, type=int)
    parser.add_argument('--lr', dest='lr',
                        help='learning rate',
                        default=0.0002, type=float)
    args = parser.parse_args()
    return args


def checkpoints_path():
    return os.path.abspath(os.path.join(os.getcwd(), "checkpoints",
                                        "AlexNet{0}.pth".format(time.strftime("%Y%m%d%H%M", time.localtime()))))


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
    args = parse_args()
    print('Called with args:')
    print(args)

    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=data_transform['train'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=data_transform['val'])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    net = alexnet(True, pretrained_path, num_classes=args.num_classes)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(args.num_epochs):
        net.train()
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^5.2f}%[{}->{}]{:.3f}".format(rate * 100, a, b, loss), end="")
        print()
        print(time.perf_counter() - t1)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / (len(val_loader)*args.batch_size)
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), checkpoints_path())
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, val_accurate))

    print('Finished Training')
