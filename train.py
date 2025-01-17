import torch
import torch.nn as nn
import torch.optim as optim
from data.datasets.ACDC.ACDCDataset import ACDCDataset
from data.Dataloader import get_dataloader
from model.LeNet import LeNet
from model.FCN import FCN32s
from model.encoder.pvt import pvt_v2_b2, model_update
import sys


def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

def trainer_one_epoch(model, epoch, train_dataloader, optimizer, loss):
    for i in range(epoch):
        model.train()
        for index, [features, labels] in enumerate(train_dataloader):
            # load data in gpu
            features, labels = features.cuda(), labels.cuda()

            output = model()

            predict = torch.argmax(output, dim=1).float()

            l = loss(output, labels)
            print(f"epoch: {i}, loss: {l}")
            l.backward()
            optimizer.step()
            optimizer.zero_grad()



def trainer(model):
    # model.apply(init_weights)

    epoch = 200
    # train_dataloader, test_dataloader = load_datasets()
    train_dataset, test_dataset = ACDCDataset("../../../datasets/ACDC/train"), ACDCDataset("../../../datasets/ACDC/test")
    train_dataloader = get_dataloader(train_dataset, batch_size=4)

    optimizer = optim.AdamW(params=model.parameters(), lr=0.001, weight_decay=0.0001)
    loss = nn.CrossEntropyLoss()
    trainer_one_epoch(model, epoch, train_dataloader, optimizer, loss)



if __name__ == "__main__":
    checkpoint_path = ''
    #model = LeNet(in_channels=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FCN32s(model_update(backbone=pvt_v2_b2(), checkpoint_path=''), n_class=5)
    model.cuda()
    trainer(model=model)
