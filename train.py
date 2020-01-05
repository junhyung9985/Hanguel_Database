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
from PIL import Image
import csv
from os import walk
from os.path import isfile, join
from os import listdir
import glob
import warnings
warnings.filterwarnings("ignore")

'''

train_path = "../Hanguel_Database/Image_train"
test_path = "../Hanguel_Database/Image_test"
#ts = "/content/drive/My Drive/DeepLearning_group3/Hanguel_Database/Image_test"

def findFiles(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

test_file = findFiles(test_path)
train_file = findFiles(train_path)


test_dir = "../Hanguel_Database/test"
train_dir = "../Hanguel_Database/train"

label = open('labels.csv', 'r', encoding='utf-8')
rdr = csv.reader(label)
for line in rdr:
    line = line[0]
    save_ts = test_dir+"/"+line
    save_tr = train_dir+"/"+line
    if (os.path.exists(save_ts)==False): os.makedirs(os.path.join(save_ts))
    if (os.path.exists(save_tr)==False): os.makedirs(os.path.join(save_tr))

for f in train_file:
    fi = f.split('_')
    class_name = fi[0]
    num = (fi[1].split('.'))[0]
    ls = list(csv.reader(open(join(train_path, f), newline=''), delimiter=','))
    lst = []
    for tmp in ls:
        temp = [a for a in tmp if a!='']
        lst.append(temp)
    print(lst)
    matrix = np.array(lst).astype("uint8")
    imgObj = Image.fromarray(matrix)
    resized_imgObj = imgObj.resize((28, 28))
    resized_imgObj.save("../Hanguel_Database/train/{}/{}.jpg".format(class_name,num))
'''
# Training_set을 ImageLoader와 호환할 수 있게 해주는 코드
'''

train_path = "../Hanguel_Database/Image_train"
test_path = "../Hanguel_Database/Image_test"
#ts = "/content/drive/My Drive/DeepLearning_group3/Hanguel_Database/Image_test"

def findFiles(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles

test_file = findFiles(test_path)
train_file = findFiles(train_path)

test_dir = "../Hanguel_Database/test"
train_dir = "../Hanguel_Database/train"

label = open('labels.csv', 'r', encoding='utf-8')
rdr = csv.reader(label)
for line in rdr:
    line = line[0]
    save_ts = test_dir+"/"+line
    save_tr = train_dir+"/"+line
    if (os.path.exists(save_ts)==False): os.makedirs(os.path.join(save_ts))
    if (os.path.exists(save_tr)==False): os.makedirs(os.path.join(save_tr))

for f in test_file:
    fi = f.split('_')
    class_name = fi[0]
    num = (fi[1].split('.'))[0]
    ls = list(csv.reader(open(join(test_path, f), newline=''), delimiter=','))
    lst = []
    for tmp in ls:
        temp = [a for a in tmp if a!='']
        lst.append(temp)
    print(lst)
    matrix = np.array(lst).astype("uint8")
    imgObj = Image.fromarray(matrix)
    resized_imgObj = imgObj.resize((28, 28))
    resized_imgObj.save("../Hanguel_Database/test/{}/{}.jpg".format(class_name,num))


'''
# Test_set을 ImageLoader와 호환할 수 있게 해주는 코드

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

