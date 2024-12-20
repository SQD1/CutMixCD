import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
# from base import BaseModel
# from utils.helpers import set_trainable
# from utils.losses import *
from models.decoders import *
from models.encoder import Encoder
# from utils.losses import CE_loss

class ResNet50_CD(nn.Module):
    def __init__(self, num_classes, pretrained=None):
        super(ResNet50_CD, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        # create the model
        self.encoder = Encoder(pretrained=pretrained)
        upscale = 8
        num_out_ch = 2048
        decoder_in_ch = num_out_ch // 4
        self.decoder = MainDecoder(upscale, decoder_in_ch, num_classes=num_classes)

    def forward(self, x, return_features=False):
        if return_features:  # return change predictions and features
            features = self.encoder(x[0], x[1])
            return self.decoder(features), features
        else:
            return self.decoder(self.encoder(x[0], x[1]))

    def pretrained_parameters(self):
        if self.pretrained:
            return list(self.encoder.get_backbone_params())
        else:
            return []

    def new_parameters(self):
        if self.pretrained:
            pretrained_ids = [id(p) for p in self.encoder.get_backbone_params()]
            return [p for p in self.parameters() if id(p) not in pretrained_ids]
        else:
            return list(self.parameters())

# net = ResNet50_CD(num_classes=2, pretrained=None)
# A = torch.rand([4,3,256,256])
# B = torch.rand([4,3,256,256])
# out = net(A,B)
# print(out.shape)




