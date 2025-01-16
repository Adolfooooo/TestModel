import torch
import torch.nn as nn
import torch.optim as optim
from data.datasets.ACDC.ACDCDataset import ACDCDataset
from data.Dataloader import get_dataloader


import sys


class LeNet(nn.Module):
    def __init__(self, in_channels, out_channels_list=[6, 12], kernel_size=5, stride=1, padding=2):
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
        
        return self.fc(front)



def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)


def trainer(model):
    # model.apply(init_weights)

    epoch = 200
    # train_dataloader, test_dataloader = load_datasets()
    train_dataset, test_dataset = ACDCDataset("./datasets/ACDC/train"), ACDCDataset("./datasets/ACDC/test")
    train_dataloader = get_dataloader(train_dataset, batch_size=4)

    optimizer = optim.AdamW(params=model.parameters(), lr=0.001, weight_decay=0.0001)
    loss = nn.CrossEntropyLoss()

    for i in range(epoch):
        model.train()
        for index, [features, labels] in enumerate(train_dataloader):
            features, labels = features.cuda(), labels.cuda()
            output = model(features)
            predict = torch.argmax(output, dim=1).float()
            print(output.shape, predict.shape, labels.shape)
            print(predict)
            l = loss(output, labels)
            print(f"epoch: {i}, loss: {l}")
            l.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    model = LeNet(in_channels=1)
    model.cuda()
    trainer(model=model)
