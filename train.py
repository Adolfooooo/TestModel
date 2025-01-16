import torch
import torch.nn as nn
import torch.optim as optim
from data.datasets.ACDC.ACDCDataset import ACDCDataset
from data.Dataloader import get_dataloader
from model.LeNet import LeNet
from model.FCN import FCN32s
import sys


def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)


def trainer(model):
    # model.apply(init_weights)

    epoch = 200
    # train_dataloader, test_dataloader = load_datasets()
    train_dataset, test_dataset = ACDCDataset("../../../datasets/ACDC/train"), ACDCDataset("../../../datasets/ACDC/test")
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
    #model = LeNet(in_channels=1)
    model = FCN32s(pretrained_net = nn.Conv2d(1, 512, kernel_size=1), n_class=5)
    model.cuda()
    trainer(model=model)
