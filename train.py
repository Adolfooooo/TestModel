import torch
import torch.nn as nn
import torch.optim as optim
from data.datasets.ACDC.ACDCDataset import ACDCDataset
from data.Dataloader import get_dataloader
from model.LeNet import LeNet
from model.FCN import FCN32s
from model.encoder.pvt import pvt_v2_b2, model_update
from utils.log import get_logger
from utils.metric import IOUMetric
from utils.pytorch_visualization import visualize_feature_maps, visualize_labels_feature_maps

import sys


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train_epochs(model, epoch, train_dataloader, optimizer, loss, num_classes, logger):
    for i in range(epoch):
        model.train()

        iou_metric = IOUMetric(num_classes=num_classes)
        for index, [features, labels] in enumerate(train_dataloader):
            # load data in gpu
            features, labels = features.cuda(), labels.cuda()

            output = model(features)
            
            predict = torch.argmax(output, dim=1).float()

            l = loss(output, labels)
            print(f"epoch: {i}, loss: {l}")
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            iou_metric.add_batch(predict.detach().cpu().numpy(), labels.detach().cpu().numpy())
        with torch.no_grad():
            acc, acc_cls, iu, mean_iu, fwavacc = iou_metric.evaluate()
            logger.info(f"acc: {acc}, acc_cls: {acc_cls}, iu: {iu}, mean_iu: {mean_iu}, fwavacc:{fwavacc}")
            

def trainer(model, logger):
    # model.apply(init_weights)

    epoch = 100
    # train_dataloader, test_dataloader = load_datasets()
    train_dataset, test_dataset = ACDCDataset("../ACDC/train"), ACDCDataset("../ACDC/test")
    train_dataloader = get_dataloader(train_dataset, batch_size=4)

    optimizer = optim.AdamW(params=model.parameters(), lr=0.001, weight_decay=0.0001)
    loss = nn.CrossEntropyLoss()
    train_epochs(model, epoch, train_dataloader, optimizer, loss, num_classes=4, logger=logger)

    # save model, and there are save two kinds of model
    torch.save(model, 'whole_model.pth')
    torch.save(model.state_dict(), 'model_params.pth')


if __name__ == "__main__":
    checkpoint_path = '../pvt_v2_b2.pth'
    logger = get_logger("pvt_fcn_output_224_512")
    #model = LeNet(in_channels=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FCN32s(model_update(backbone=pvt_v2_b2(), checkpoint_path=checkpoint_path), n_class=4)
    model.cuda()
    trainer(model=model, logger=logger)
