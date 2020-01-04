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
        super(MyCNN, self).__init__()

        self.output_dim = output_dim

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # try with different kernels
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 x (16x16)

            nn.Conv2d(32, 16, 2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16 x (8x8)

            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 8 x (4x4)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, output_dim)
        )

    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.view(out.shape[0], -1)
        out = self.fc_layer(out)

        return out


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(32,32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.3, 0.3, 0.3])
    ]),
    'test': transforms.Compose([
        transforms.Resize(32,32),
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
model_name = input("Input(model_name): ")

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
