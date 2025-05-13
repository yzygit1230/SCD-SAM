import torch
import torch.nn as nn
from models.block.Base import Conv3Relu
from models.block.Drop import DropBlock


class FPNNeck(nn.Module):
    def __init__(self, inplanes, neck_name='fpn+ppm+fuse'):
        super().__init__()
        self.stage1_Conv1 = Conv3Relu(inplanes * 2, inplanes)  
        self.stage2_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2)  
        self.stage3_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)  
        self.stage4_Conv1 = Conv3Relu(inplanes * 16, inplanes * 8)  

        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)

        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.d2 = Conv3Relu(inplanes * 2, inplanes)
        self.d3 = Conv3Relu(inplanes * 4, inplanes)
        self.d4 = Conv3Relu(inplanes * 8, inplanes)

        if "drop" in neck_name:
            rate, size, step = (0.15, 7, 30)
            self.drop = DropBlock(rate=rate, size=size, step=step)
        else:
            self.drop = DropBlock(rate=0, size=0, step=0)

    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])  

        change1 = self.stage1_Conv1(torch.cat([fa1, fb1], 1)) 
        change2 = self.stage2_Conv1(torch.cat([fa2, fb2], 1)) 
        change3 = self.stage3_Conv1(torch.cat([fa3, fb3], 1))  
        change4 = self.stage4_Conv1(torch.cat([fa4, fb4], 1))  

        change3_2 = self.stage4_Conv_after_up(self.up(change4))
        change3 = self.stage3_Conv2(torch.cat([change3, change3_2], 1))
        change2_2 = self.stage3_Conv_after_up(self.up(change3))
        change2 = self.stage2_Conv2(torch.cat([change2, change2_2], 1))
        change1_2 = self.stage2_Conv_after_up(self.up(change2))
        change1 = self.stage1_Conv2(torch.cat([change1, change1_2], 1))

        return change1

