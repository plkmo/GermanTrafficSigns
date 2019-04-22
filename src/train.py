# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:27:50 2019

@author: WT
"""
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

def load_pickle(filename):
    completeName = os.path.join("C:/Users/WT/Desktop/Python_Projects/TrafficLights/data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

class trafficdata(Dataset):
    def __init__(self, images, labels, transform=None):
        super(trafficdata, self).__init__()
        self.transform = transform
        self.X = images
        self.y = torch.tensor(labels,requires_grad=False)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        if self.transform:
            Xidx = self.transform(self.X[idx])
        else:
            Xidx = self.X[idx]
        return Xidx, self.y[idx]
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(64,64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            #3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 42))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

### Loads model and optimizer states
def load(net, optimizer, load_best=False):
    base_path = "C:/Users/WT/Desktop/Python_Projects/TrafficLights/data/"
    if load_best == False:
        checkpoint = torch.load(os.path.join(base_path,"checkpoint.pth.tar"))
    else:
        checkpoint = torch.load(os.path.join(base_path,"model_best.pth.tar"))
    start_epoch = checkpoint['epoch']
    best_pred = checkpoint['best_acc']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return start_epoch, best_pred

def model_eval(net, test_loader, cuda=None):
    correct = 0
    total = 0
    with torch.no_grad():
        net.eval()
        for data in test_loader:
            images, labels = data
            if cuda:
                images, labels = images.cuda(), labels.cuda()
            images = images.float(); labels = labels.long()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print("Accuracy of the network on the %d test images: %d %%" % (total,\
                                                                    100*correct/total))
    return 100*correct/total

if __name__ == "__main__":
    train_images = load_pickle("train_images.pkl")
    train_labels = load_pickle("train_images_labels.pkl")
    
    test_images = load_pickle("test_images.pkl")
    test_labels = load_pickle("test_images_labels.pkl")
    
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),\
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([transforms.ToTensor(),\
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 50
    trainset = trafficdata(train_images, train_labels, transform=transform)
    testset = trafficdata(test_images, test_labels, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
    cuda = torch.cuda.is_available()
    net = Net()
    if cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    try:
        start_epoch, best_pred = load(net, optimizer, load_best=True)
    except:
        start_epoch = 0; best_pred = 0
    end_epoch = 30
    losses_per_epoch = []; accuracy_per_epoch = []
    
    for epoch in range(start_epoch, end_epoch):
        net.train(); total_loss = 0.0; losses_per_batch = []
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs = inputs.float(); labels = labels.long()
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches of size = batch_size
                print('[Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                      (epoch + 1, (i + 1)*batch_size, len(trainset), total_loss/100))
                losses_per_batch.append(total_loss/100)
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        score = model_eval(net, test_loader, cuda=cuda)
        accuracy_per_epoch.append(score)
        if score > best_pred:
            best_pred = score
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': score,\
                    'optimizer' : optimizer.state_dict(),\
                }, os.path.join("C:/Users/WT/Desktop/Python_Projects/TrafficLights/data/" ,"model_best.pth.tar"))
        if (epoch % 10) == 0:
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'best_acc': score,\
                    'optimizer' : optimizer.state_dict(),\
                }, os.path.join("C:/Users/WT/Desktop/Python_Projects/TrafficLights/data/" ,"checkpoint.pth.tar"))
    
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.scatter([e for e in range(start_epoch,end_epoch,1)], losses_per_epoch)
    ax.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax.set_xlabel("Epoch", fontsize=22)
    ax.set_ylabel("Loss per batch", fontsize=22)
    ax.set_title("Loss vs Epoch", fontsize=32)
    print('Finished Training')
    plt.savefig(os.path.join("C:/Users/WT/Desktop/Python_Projects/TrafficLights/data/" ,"loss_vs_epoch.png"))
    
    fig2 = plt.figure(figsize=(20,20))
    ax2 = fig2.add_subplot(111)
    ax2.scatter([e for e in range(start_epoch,end_epoch,1)], accuracy_per_epoch)
    ax2.tick_params(axis="both", length=2, width=1, labelsize=14)
    ax2.set_xlabel("Epoch", fontsize=22)
    ax2.set_ylabel("Test Accuracy", fontsize=22)
    ax2.set_title("Test Accuracy vs Epoch", fontsize=32)
    print('Finished Training')
    plt.savefig(os.path.join("C:/Users/WT/Desktop/Python_Projects/TrafficLights/data/" ,"accuracy_vs_epoch.png"))