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
from utils.pytorch_visualization import visualize_feature_maps, visualize_non_channel_feature_maps

import sys


if __name__ == "__main__":
    num_classes=4
    checkpoint_path = '../pvt_v2_b2.pth'
    logger = get_logger("pvt_fcn_output_224_512")
    #model = LeNet(in_channels=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load("whole_model.pth")
    model.cuda()
    train_dataset, test_dataset = ACDCDataset("../ACDC/train"), ACDCDataset("../ACDC/test")
    train_dataloader = get_dataloader(train_dataset, batch_size=1)
    for index, [features, labels] in enumerate(train_dataloader):
        # load data in gpu
        features, labels = features.cuda(), labels.cuda()
        
        visualize_feature_maps(features)
        visualize_non_channel_feature_maps(labels)
        output = model(features)
        predict = torch.argmax(output, dim=1).float()
        
        visualize_feature_maps(predict, pic_name="predict", is_batch=False)
        iou_metric = IOUMetric(num_classes=num_classes)
        iou_metric.add_batch(predict.detach().cpu().numpy(), labels.detach().cpu().numpy())
        acc, acc_cls, iu, mean_iu, fwavacc = iou_metric.evaluate()
        import sys
        sys.exit()

