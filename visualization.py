import os
import time
import argparse
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.common import ScaleInOutput
from PIL import Image
import torch
import numpy as np
#################################
from datasets import RS_ST as RS
from main_model import ChangeDetection

#################################
NET_NAME = 'SAMNet'

# DATA_NAME = 'SECOND'
DATA_NAME = 'Landsat'

scale = ScaleInOutput(512)

class PredOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        TEST_DIR = 'datasets/SECOND/val/'
        PRED_DIR = 'datasets/SECOND_res_set/' + NET_NAME + '/'
        parser.add_argument('--pred_batch_size', required=False, default=1, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default=TEST_DIR, help='directory to test images')
        parser.add_argument('--pred_dir', required=False, default=PRED_DIR, help='directory to output masks')
        parser.add_argument('--pth_path', required=False, default=working_path + ''
        # '/checkpoints/'+DATA_NAME+'/'+NET_NAME+'/SAMNet_93e_mIoU77.75_Sek32.44_Fscd70.51_OA90.16.pth')
        '/checkpoints/'+DATA_NAME+'/'+NET_NAME+'/SAMNet_70e_mIoU83.04_Sek44.05_Fscd81.16_OA93.30.pth')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt


def main(net_opt):
    begin_time = time.time()
    opt = PredOptions().parse()
    net = ChangeDetection(net_opt).cuda()
    net.load_state_dict(torch.load(opt.pth_path), strict=False)
    net.eval()
    test_set = RS.Data_test(opt.test_dir)
    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    predict(net, test_set, test_loader, opt.pred_dir, flip=False, index_map=True, intermediate=False)
    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)

def predict(net, pred_set, pred_loader, pred_dir, flip=False, index_map=False, intermediate=False):
    pred_A_dir_rgb = os.path.join(pred_dir, 'im1_rgb')
    pred_B_dir_rgb = os.path.join(pred_dir, 'im2_rgb')
    if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
    if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)

    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B, la,lb = data

        imgs_A = imgs_A.cuda().float()
        imgs_B = imgs_B.cuda().float()
        mask_name = pred_set.get_mask_name(vi)

        with torch.no_grad():
            imgs_A, imgs_B = scale.scale_input((imgs_A, imgs_B))
            outputs_A, outputs_B, out_change = net(imgs_A, imgs_B)  
            out_change = F.sigmoid(out_change)

        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = out_change.cpu().detach() > 0.5
        change_mask = change_mask.squeeze()
        pred_A = torch.argmax(outputs_A, dim=1).squeeze()
        pred_B = torch.argmax(outputs_B, dim=1).squeeze()

        pred_A = (pred_A * change_mask.long()).numpy()
        pred_B = (pred_B * change_mask.long()).numpy()
        pred_A = Image.fromarray(RS.Index2Color(pred_A).astype('uint8'))
        pred_B = Image.fromarray(RS.Index2Color(pred_B).astype('uint8'))
        
        pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
        pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)

        pred_A.save(pred_A_path, quality=95, dpi=(300, 300))
        pred_B.save(pred_B_path, quality=95, dpi=(300, 300))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Change Detection train')
    parser.add_argument("--backbone", type=str, default="msam_96")
    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")
    parser.add_argument("--pretrain", type=str, default="") 
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.00035)
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    net_opt = parser.parse_args()

    main(net_opt)