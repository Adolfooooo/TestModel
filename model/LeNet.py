from torch import nn

class LeNet(nn.Module):
    def __init__(self, in_channels, out_channels_list=[6, 12, 24], kernel_size=5, stride=1, padding=2):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels=out_channels_list[0], kernel_size=kernel_size, stride=stride, padding=padding),
            # nn.BatchNorm2d(num_features=out_channels_list[0]),
            nn.Sigmoid(),
        )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels_list[0], out_channels=out_channels_list[1], kernel_size=kernel_size),
            # nn.BatchNorm2d(num_features=out_channels_list[1]),
            nn.Sigmoid(),
        )
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels_list[0], out_channels=out_channels_list[1], kernel_size=kernel_size),
            # nn.BatchNorm2d(num_features=out_channels_list[1]),
            nn.Sigmoid(),
        )
        self.avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(12*5*5, out_features=120),
            nn.Sigmoid(),
            nn.Linear(120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(84, out_features=10)
        )
        
    
    def forward(self, x):
        front = self.avgpool2(self.conv2(self.avgpool1(self.conv1(x))))
        
        front = self.avgpool3(self.conv3)
        return self.fc(front)