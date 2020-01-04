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
test_batch_size = 10
data_dir = '/home/junhyung9985/Hanguel_Database/'
test_dir = 'test'
test_set = datasets.ImageFolder(test_dir, test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=4)


result_dir = '/home/junhyung9985/Hanguel_Database/result/'
model_name = 'team3.model'

model = torch.load(result_dir + model_name)
model.cuda()
model.eval()
hits = 0

for k, [image, label] in enumerate(test_loader):
    x = image.cuda()
    y_ = label.cuda()

    output = model(x)
    y_est = output.argmax(1)
    print('Target', label.numpy(), 'Prediction', y_est.cpu().numpy())
    hits = hits + sum(y_est == y_)
print('hits, accuracy', hits, hits / (len(test_set) + 0.0))
