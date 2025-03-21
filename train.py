import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim

from data.datasets.ACDC.ACDCDataset import ACDCDataset
from data.Dataloader import get_dataloader
from data.datasets.Synapse.dataset_synapse import Synapse_dataset, RandomGenerator

from model.LeNet import LeNet
from model.FCN import FCN32s
from model.UNet import unet_pvtb2, resnet34_unet
from model.encoder.pvt import pvt_v2_b2, model_update

from utils.log import get_logger
from utils.metric import IOUMetric
from utils.gradient_monitor import GradientMonitor
from utils.global_variable import global_variable_init,global_variable_set_dict,global_variable_get
from utils.initialize_model import ModelInitializer
from utils.losses import DiceLoss

import sys


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def train_epochs(model, epoch, train_dataloader, optimizer, loss, num_classes, logger):
    monitor = GradientMonitor(model)
    for i in range(epoch):
        model.train()

        iou_metric = IOUMetric(num_classes=num_classes)
        for index, [features, labels] in enumerate(train_dataloader):
            # load data in gpu
            features, labels = features.cuda(), labels.cuda()

            # output shape: [B, C, H , W]
            output = model(features)

            l = loss(output, labels)
            logger.info(f"epoch: {i}, loss: {l}")

            # monitor.step_monitor()

            # # 每隔一定步数绘制分布图
            # if monitor.step % 100 == 0:
            #     monitor.plot_gradient_distribution()
            #     monitor.plot_gradient_flow()
            
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
            # iou_metric.add_batch(predict.detach().cpu().numpy(), labels.detach().cpu().numpy())
        # with torch.no_grad():
            # acc, acc_cls, iu, mean_iu, fwavacc = iou_metric.evaluate()
            # logger.info(f"acc: {acc}, acc_cls: {acc_cls}, iu: {iu}, mean_iu: {mean_iu}, fwavacc:{fwavacc}")

    # monitor.close()
            

def trainer(model, logger, num_classes):
    # model.apply(init_weights)
    num_classes = 4
    epoch = 50
    # train_dataloader, test_dataloader = load_datasets()
    train_dataset, test_dataset = ACDCDataset("../ACDC/train"), ACDCDataset("../ACDC/test")
    # train_dataset = Synapse_dataset(base_dir="../synapse/train_npz", list_dir="../synapse/lists/lists_Synapse", split="train", nclass=num_classes, 
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[224, 224])]))
    train_dataloader = get_dataloader(train_dataset, batch_size=4)

    optimizer = optim.AdamW(params=model.parameters(), lr=0.001, weight_decay=0.001)
    # loss = CrossEntropyLoss()
    # loss = nn.BCELoss()
    loss = DiceLoss(num_classes)
    train_epochs(model, epoch, train_dataloader, optimizer, loss, num_classes=num_classes, logger=logger)

    # save model, and there are save two kinds of model
    torch.save(model, 'whole_model.pth')
    torch.save(model.state_dict(), 'model_params.pth')
    

if __name__ == "__main__":
    checkpoint_path = '../pvt_v2_b2.pth'
    logger = get_logger("resnet34_unet_output_224_512")
    global_variable_init()
    global_variable_set_dict("logger", logger)
    initializer = ModelInitializer()
        
    # hyper param
    num_classes = 4 # ACDC

    #model = LeNet(in_channels=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = FCN32s(model_update(backbone=pvt_v2_b2(), checkpoint_path=checkpoint_path), n_class=4)
    # model = unet_pvtb2(model_update(backbone=pvt_v2_b2(), checkpoint_path=checkpoint_path), num_classes=4)
    model = resnet34_unet(num_classes=num_classes)
    initializer.xavier_uniform_init(model)
    model.cuda()
    trainer(model=model, logger=logger, num_classes=num_classes)
