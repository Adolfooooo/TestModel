from torch import nn
from utils.global_variable import global_variable_set_dict,global_variable_get

class FCN32s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        self.input_preprocess = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1, stride=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
    def forward(self, x):
        #print(x.shape)
        x = self.input_preprocess(x)
                                 
        # logger = global_variable_get("logger")
        # specify_dim_size = x.shape[1]
        # for index in range(specify_dim_size):
        #     # logger.info(f"specify_dim: channel {index}: {score[0][index]}")
        #     logger.info(f"specify_dim: channel {index}: {x[0][index].sum()}")
        # logger.info("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

        
        x1, x2, x3, x4 = self.pretrained_net(x)

        
        # print(output['x1'].shape)
        # print(output['x2'].shape)
        # print(output['x3'].shape)
        # print(output['x4'].shape)
#        x4 = output['x4']  # size=(N, 512, x.H/32, x.W/32)
        #print(x5.shape)
        score = self.bn1(self.relu(self.deconv1(x4)))     # size=(N, 512, x.H/16, x.W/16)

        
        
        #print(score.shape)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        #print(score.shape)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        #print(score.shape)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        #print(score.shape)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)

        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        
        
        # logger.info(f"score: channel 0:{score[0][0]}")
        # logger.info(f"score: channel 1:{score[0][1]}")
        # logger.info(f"score: channel 2:{score[0][2]}")
        # logger.info(f"score: channel 3:{score[0][3]}\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        score = self.sigmoid(score)
        # print(score.shape)
        
        
        return score  # size=(N, n_class, x.H/1, x.W/1)
