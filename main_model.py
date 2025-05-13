import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from models.block.Base import ChannelChecker
from models.block.Base import Conv1Relu
from models.Encoder.mobilesam import build_sam_vit_t
from models.Encoder.moat import moat_4
from models.Encoder.DFI import *
from models.Decoder.FPN import FPNNeck
from models.Decoder.AFPN_CARAFE import AFPN
from models.head.CSD import *
from models.head.FCN import *

class ChangeDetection(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = int(re.sub(r"\D", "", opt.backbone.split("_")[-1]))  
        self._create_backbone(opt.backbone)
        self._create_neck(opt.neck)
        self._create_heads(opt.head)
        if opt.pretrain.endswith(".pt"):
            self._init_weight(opt.pretrain)   
        self.check_channels = ChannelChecker(self.backbone, self.inplanes, opt.input_size)
        self.fusion4 = Transformer(self.inplanes*8, self.inplanes*8, 16, 16)
        self.conv4 = Conv1Relu(self.inplanes*16, self.inplanes*8)


    def forward(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        fa1, fa2, fa3, _,fa4 = self.backbone(xa)  
        fa1, fa2, fa3,fa4 = self.check_channels(fa1, fa2, fa3, fa4)
        fb1, fb2, fb3, _,fb4 = self.backbone(xb)
        fb1, fb2, fb3,fb4 = self.check_channels(fb1, fb2, fb3, fb4)

        fa12, fa22, fa32, fa42 = self.backbone2(xa)
        fb12, fb22, fb32, fb42 = self.backbone2(xb)

        inentity_a4 = fa4
        inentity_b4 = fb4
        fa4 = self.conv4(torch.cat([inentity_a4, self.fusion4(fa4, fa42)], 1))
        fb4 = self.conv4(torch.cat([inentity_b4, self.fusion4(fb4, fb42)], 1))

        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4  

        change = self.neck(ms_feats)
        change_s1 = self.senmantic_neck(ms_feats)
        change_s2 = self.senmantic_neck(ms_feats)

        out1, out2, out = self.head_forward(change, change_s1, change_s2, out_size=(h_input, w_input))

        return out1, out2, out

    def head_forward(self, change, change_s1, change_s2, out_size):
        change_s1, change_s2 = self.CotSR(change_s1, change_s2)
        out, out1, out2 = self.head(change, change_s1, change_s2)
        
        out1 = F.interpolate(out1, size=out_size, mode='bilinear', align_corners=True)
        out2 = F.interpolate(out2, size=out_size, mode='bilinear', align_corners=True)
        out = F.interpolate(out, size=out_size, mode='bilinear', align_corners=True)
       
        return out1, out2, out

    def _init_weight(self, pretrain=''):  
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))

    def _create_backbone(self, backbone):
        self.backbone = build_sam_vit_t()
        self.backbone2 = moat_4(use_window=True, num_classes=10)

    def _create_neck(self, neck):
        self.neck = FPNNeck(self.inplanes, neck)
        self.senmantic_neck = AFPN(self.inplanes)
        
    def _create_heads(self, head):
        self.head = FCNHead(self.inplanes, 1)
        self.CotSR = CotSR(in_dim=self.inplanes, num_classes=self.inplanes) 
