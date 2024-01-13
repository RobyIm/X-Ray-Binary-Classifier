import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

import hyperparameters as hp

# BinaryX_RayCNN model
class BinaryX_RayCNN(nn.Module):

    """
    Custom CNN model for binary X-Ray classification with filters.

    Parameters
    ----------
    None

    Attributes
    ----------
    output_size : int
        Size of the output feature map.
    
    input_size : int
        Size of the input feature map.
    
    conv1 : nn.Conv2d
        First convolutional layer with 3 input channels, 16 output channels, and a variable-sized kernel.
    
    conv2 : nn.Conv2d
        Second convolutional layer with 16 input channels, 32 output channels, and a 1x1 kernel.
    
    conv3 : nn.Conv2d
        Third convolutional layer with 32 input channels, 3 output channels, and a 5x5 kernel.
    
    resnet : torchvision.models.resnet.ResNet
        Pre-trained ResNet50 model with modified fully connected layer for binary classification.

    Methods
    -------
    _calc_pad(kernel_size, stride=1)
        Calculates the padding needed for a convolutional layer given the kernel size and stride.

    forward(x)
        Forward pass of the model.

    """

    def __init__(self):
        super(BinaryX_RayCNN, self).__init__()
        self.output_size = 224
        self.input_size = 224

        # First convolution layer
        kernel1 = 3
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16, 
            kernel_size=kernel1, 
            stride=1, 
            padding=self._calc_pad(kernel1))
        
        # Second convolution layer
        kernel2 = 1
        self.conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=kernel2, 
            stride=1, 
            padding=self._calc_pad(kernel2))

        # Third convolution layer
        kernel3 = 5
        self.conv3 = nn.Conv2d(in_channels=32,
            out_channels=3,
            kernel_size=kernel3, 
            stride=1, 
            padding=self._calc_pad(kernel3))

        # Pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=True)
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        self.resnet.load_state_dict(state_dict)
        self.resnet.fc = nn.Linear(2048, hp.NUM_CLASSES)

    def _calc_pad(self, kernel_size, stride=1):

        """
        Calculates the padding needed for a convolutional layer given the kernel size and stride.

        Parameters
        ----------
        kernel_size : int
            Size of the convolutional kernel.
        
        stride : int, optional
            Stride of the convolution operation. Default is 1.

        Returns
        -------
        int
            Padding required for the convolutional layer.

        """

        return (((self.output_size-1) * stride) + 1 + kernel_size - 1 - self.input_size) // 2

    def forward(self, x):

        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after passing through the model.

        """
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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