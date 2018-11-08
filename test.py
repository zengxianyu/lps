# coding=utf-8
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import pdb
import numpy as np
from PIL import Image
import argparse
import json

from datasets import PriorFolder, Folder
from datasets.saliency import collate_more
from models import Net, FCN
from evaluate import fm_and_mae

from tqdm import tqdm
import random

random.seed(1996)


home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--prior_dir', default='%s/data/datasets/saliency_Dataset/results/HKU-IS-Sal/SRM' % home)  # training dataset
parser.add_argument('--val_dir', default='%s/data/datasets/saliency_Dataset/HKU-IS' % home)  # training dataset
parser.add_argument('--base', default='vgg16')  # training dataset
parser.add_argument('--img_size', type=int, default=256)  # batch size
parser.add_argument('--b', type=int, default=12)  # batch size
parser.add_argument('--max', type=int, default=100000)  # epoches
opt = parser.parse_args()
print(opt)

name = 'Train_{}'.format(opt.base)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# tensorboard writer
os.system('rm -rf ./runs_%s/*'%name)
writer = SummaryWriter('./runs_%s/'%name + datetime.now().strftime('%B%d  %H:%M:%S'))
if not os.path.exists('./runs_%s'%name):
    os.mkdir('./runs_%s'%name)


def make_image_grid(img, mean, std):
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img


def validate(loader, net, output_dir, gt_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    net.eval()
    loader = tqdm(loader, desc='validating')
    for ib, (data, lbl, prior, img_name, w, h) in enumerate(loader):
        with torch.no_grad():
            outputs = net(data.cuda(), prior[:, None].cuda())
            outputs = F.sigmoid(outputs)
        outputs = outputs.squeeze(1).cpu().numpy()
        outputs *= 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((w[ii], h[ii]))
            msk.save('{}/{}.png'.format(output_dir, img_name[ii]), 'PNG')
    fm, mae, _, _ = fm_and_mae(output_dir, gt_dir)
    pfm, pmae, _, _ = fm_and_mae(opt.prior_dir, gt_dir)
    net.train()
    print('%.4f, %.4f'%(pfm, pmae))
    print('%.4f, %.4f'%(fm, mae))
    return fm, mae


def main():

    check_dir = '../LPSfiles/' + name

    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    # models
    net = Net(base=opt.base)
    net = nn.DataParallel(net).cuda()
    sdict =torch.load('../LPSfiles/Train_vgg16/net.pth')
    net.load_state_dict(sdict)
    val_loader = torch.utils.data.DataLoader(
        PriorFolder(opt.val_dir, opt.prior_dir, size=256,
                    mean=mean, std=std),
        batch_size=opt.b, shuffle=False, num_workers=4, pin_memory=True)
    fm, mae = validate(val_loader, net, os.path.join(check_dir, 'results'),
                                      os.path.join(opt.val_dir, 'masks'))


if __name__ == "__main__":
    main()