import torch.nn as nn
from models.block.Base import Conv3Relu
import torch

class CotSR(nn.Module):

    def __init__(self, in_dim=96, num_classes=96):
        super(CotSR, self).__init__()
        self.in_dim = in_dim
        self. num_classes= num_classes
     
        self.query_conv1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim //8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim//8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=1)
        
        self.query_conv2 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim //8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim //8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=self.in_dim, out_channels=self.in_dim, kernel_size=1)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.classifier1 = nn.Conv2d(self.in_dim, self. num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(self.in_dim, self. num_classes, kernel_size=1)
    
    def forward(self, x1, x2):
        m_batchsize, C, height, width = x1.size()
        
        q1 = self.query_conv1(x1).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width*height)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width*height)
        
        q2 = self.query_conv2(x2).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width*height)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width*height)
        
        energy1 = torch.bmm(q1, k2)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v2, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)
                
        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        
        out1 = x1 + self.gamma1*out1
        out2 = x2 + self.gamma2*out2  

        out1 = self.classifier1(out1)
        out2 = self.classifier2(out2)  
        
        return out1, out2

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = in_channels
        self.head1 = nn.Sequential(Conv3Relu(in_channels, inter_channels),
                                  nn.Dropout(0.2), 
                                  nn.Conv2d(inter_channels, 5, (1, 1)))
        self.head2 = nn.Sequential(Conv3Relu(in_channels, inter_channels),
                                  nn.Dropout(0.2), 
                                  nn.Conv2d(inter_channels, 5, (1, 1)))
        self.head3 = nn.Sequential(Conv3Relu(in_channels, inter_channels),
                                  nn.Dropout(0.2),  
                                  nn.Conv2d(inter_channels, out_channels, (1, 1)))

    def forward(self, change, change_s1, change_s2):
        return self.head3(change), self.head2(change_s1), self.head1(change_s2)

