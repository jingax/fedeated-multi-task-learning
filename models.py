import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    A residual block (He et al. 2016).
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dropout=0):
        """Constructor.
    
        Arguments:
            -in_channels: Number of input channels
            -out_channels: Number of class (for the output dimention)
            -kernel_size: Size of square filters.
            -padding: Padding size.
            -stride: Stride amplitude.
            -dropout: Percentage of neurons to drop.
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.dropout1 = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.dropout2 = nn.Dropout2d(dropout)
        if stride != 1:
            # Downsample residual in case stride > 1
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        if self.downsample is not None:
            res = self.downsample(res)
        x = self.relu(x)
        x = x + res
        return x

class ResNet9(nn.Module):
    """
    Residual network with 9 layers.
    """
    def __init__(self, in_channels, feat_dim, output_shape, dropout=0):
        """Constructor.
    
        Arguments:
            -in_channels: Number of input channels
            -feat_dim: Feature (last hidden layer) dimension.
            -output_shape: Number of class (for the output dimention)
            -dropout: Percentage of neurons to drop.
        """
        super(ResNet9, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dropout=dropout),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dropout=dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=256, out_features=feat_dim),
            nn.Tanh(),
            nn.Dropout(dropout))
                                      
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=output_shape, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=output_shape, out_features=1, bias=True),
            nn.Sigmoid())
            # nn.Sigmoid()
        

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
# model = ResNet9(in_channels=3, feat_dim=100, output_shape=50)
# print(model)


