# coding=utf-8
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import pdb
import numpy as np
from PIL import Image
import argparse

from datasets import ImageFiles
from models import Net
from evaluate import fm_and_mae

from tqdm import tqdm
import random

random.seed(1996)


home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--prior_dir', default='%s/data/datasets/saliency_Dataset/results/ECSSD-Sal/SRM' % home)  # prior maps
parser.add_argument('--img_dir', default='%s/data/datasets/saliency_Dataset/ECSSD/images' % home)  # images
parser.add_argument('--gt_dir', default='%s/data/datasets/saliency_Dataset/ECSSD/masks' % home)  # ground truth
parser.add_argument('--base', default='vgg16')  # training dataset
parser.add_argument('--img_size', type=int, default=256)  # image size
parser.add_argument('--b', type=int, default=12)  # batch size
opt = parser.parse_args()
print(opt)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def validate(loader, net, output_dir, gt_dir=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    net.eval()
    loader = tqdm(loader, desc='validating')
    for ib, (data, prior, img_name, w, h) in enumerate(loader):
        with torch.no_grad():
            outputs = net(data.cuda(), prior[:, None].cuda())
            outputs = F.sigmoid(outputs)
        outputs = outputs.squeeze(1).cpu().numpy()
        outputs *= 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((w[ii], h[ii]))
            msk.save('{}/{}.png'.format(output_dir, img_name[ii]), 'PNG')
    if gt_dir is not None:
        fm, mae, _, _ = fm_and_mae(output_dir, gt_dir)
        pfm, pmae, _, _ = fm_and_mae(opt.prior_dir, gt_dir)
        print('%.4f, %.4f'%(pfm, pmae))
        print('%.4f, %.4f'%(fm, mae))


def main():
    # models
    net = Net(base=opt.base)
    net = nn.DataParallel(net).cuda()
    sdict =torch.load('./net.pth')
    net.load_state_dict(sdict)
    val_loader = torch.utils.data.DataLoader(
        ImageFiles(opt.img_dir, opt.prior_dir, size=256,
                    mean=mean, std=std),
        batch_size=opt.b, shuffle=False, num_workers=4, pin_memory=True)
    validate(val_loader, net, 'results', opt.gt_dir)


if __name__ == "__main__":
    main()