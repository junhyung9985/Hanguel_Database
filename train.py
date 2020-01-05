import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import warnings
warnings.filterwarnings("ignore")


class MyCNN(nn.Module):
    def __init__(self, output_dim=10):
        super(MyCNN,self).__init__()

        self.output_dim=output_dim

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 32 x (14x14)
            
            nn.Conv2d(32,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # 16 x (7x7)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(16*7*7,100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100,output_dim)
        )       
        
    def forward(self,x):
        out = self.cnn_layers(x)
        out = out.view(out.shape[0],-1)
        out = self.fc_layer(out)

        return out


learning_rate = 0.001
output_dim=2350

model = MyCNN(output_dim=output_dim).cuda()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

param_list = list(model.children())
print(param_list)


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(28,28),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.3, 0.3, 0.3])
    ]),
    'test': transforms.Compose([
        transforms.Resize(28,28),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.3, 0.3, 0.3])
    ]),
}
test_transform = data_transforms['test']


batch_size = 64
test_batch_size = 10
data_dir = '/home/junhyung9985/Hanguel_Database/'
train_dir = 'train'
test_dir = 'test'

train_set = datasets.ImageFolder(data_dir+train_dir, data_transforms['train'])
test_set = datasets.ImageFolder(test_dir, test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=4)          
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=4)

train_size = len(train_set)
test_size = len(test_set)

class_names = train_set.classes

print(class_names) 



result_dir = '/home/junhyung9985/Hanguel_Database/result'
num_epoch = 20

if not os.path.exists(result_dir):
    os.mkdir(result_dir)    
    
for i in range(num_epoch):
    model.train()
    for j, [image,label] in enumerate(train_loader):
        x = image.cuda()
        y_= label.cuda()
        
        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()
        
        if j % 30 == 0:
            print(i,j, loss.data.cpu())

print('training is done by max_epochs', num_epoch)
torch.save(model, result_dir + '/epoch_{}_lr_{}.model'.format(num_epoch, learning_rate))

