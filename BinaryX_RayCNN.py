import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

import hyperparameters as hp

# Custom CNN model with filters
class BinaryX_RayCNN(nn.Module):
    def __init__(self):
        super(BinaryX_RayCNN, self).__init__()

        # Custom filters
        self.gabor = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.sobel = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=1)
        self.laplacian = nn.Conv2d(32, 3, kernel_size=5, stride=1, padding=1)

        # Pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        self.resnet.load_state_dict(state_dict)
        self.resnet.fc = nn.Linear(2048, hp.NUM_CLASSES)

    def forward(self, x):
        x = self.gabor(x)
        x = self.sobel(x)
        x = self.laplacian(x)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x