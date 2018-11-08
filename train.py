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

# random.seed(1996)


home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='%s/data/datasets/saliency_Dataset/DUT-train' % home)  # training dataset
parser.add_argument('--prior_dir', default='%s/data/datasets/saliency_Dataset/results/ECSSD-Sal' % home)  # training dataset
parser.add_argument('--val_dir', default='%s/data/datasets/saliency_Dataset/ECSSD' % home)  # training dataset
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
    net.train()
    return fm, mae


def main():

    check_dir = '../LPSfiles/' + name

    if not os.path.exists(check_dir):
        os.mkdir(check_dir)

    # data
    val_loader = torch.utils.data.DataLoader(
        PriorFolder(opt.val_dir, opt.prior_dir, size=256,
               mean=mean, std=std),
        batch_size=opt.b*3, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        Folder(opt.train_dir, scales=[64]*3+[128, 256],
               crop=0.9, flip=True, rotate=None,
               mean=mean, std=std), collate_fn=collate_more,
        batch_size=opt.b*6, shuffle=True, num_workers=4, pin_memory=True)
    # models
    p = 5
    net = Net(base=opt.base)
    fcn = FCN(net)
    net = nn.DataParallel(net).cuda()
    net.train()
    """
    # fcn = nn.DataParallel(fcn).cuda()
    # sdict =torch.load('/home/crow/LPSfiles/Train2_vgg16/fcn-iter13800.pth')
    # fcn.load_state_dict(sdict)
    fcn = nn.DataParallel(fcn).cuda()
    fcn.train()
    optimizer = torch.optim.Adam([
        {'params': fcn.parameters(), 'lr': 1e-4},
    ])
    logs = {'best_it':0, 'best': 0}
    sal_data_iter = iter(train_loader)
    i_sal_data = 0
    for it in tqdm(range(opt.max)):
    # for it in tqdm(range(1)):
        # if it > 1000 and it % 100 == 0:
        #     optimizer.param_groups[0]['lr'] *= 0.5
        if i_sal_data >= len(train_loader):
            sal_data_iter = iter(train_loader)
            i_sal_data = 0
        data, lbls, _ = sal_data_iter.next()
        i_sal_data += 1
        data = data.cuda()
        lbls = [lbl.unsqueeze(1).cuda() for lbl in lbls]
        msks = fcn(data)
        loss = sum([F.binary_cross_entropy_with_logits(msk, lbl) for msk, lbl in zip(msks, lbls)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % 10 == 0:
            writer.add_scalar('loss', loss.item(), it)
            image = make_image_grid(data[:6], mean, std)
            writer.add_image('Image', torchvision.utils.make_grid(image), it)
            big_msk = F.sigmoid(msks[-1]).expand(-1, 3, -1, -1)
            writer.add_image('msk', torchvision.utils.make_grid(big_msk.data[:6]), it)
            big_msk = lbls[-1].expand(-1, 3, -1, -1)
            writer.add_image('gt', torchvision.utils.make_grid(big_msk.data[:6]), it)
        # if it % 100 == 0:
        if it != 0 and it % 100 == 0:
            fm, mae = validate(val_loader, fcn, os.path.join(check_dir, 'results'),
                               os.path.join(opt.val_dir, 'masks'))
            print(u'损失: %.4f'%(loss.item()))
            print(u'最大FM: iteration %d的%.4f, 这次FM: %.4f'%(logs['best_it'], logs['best'], fm))
            logs[it] = {'FM': fm}
            if fm > logs['best']:
                logs['best'] = fm
                logs['best_it'] = it
                torch.save(fcn.state_dict(), '%s/fcn-best.pth' % (check_dir))
            with open(os.path.join(check_dir, 'logs.json'), 'w') as outfile:
                json.dump(logs, outfile)
            torch.save(fcn.state_dict(), '%s/fcn-iter%d.pth' % (check_dir, it))
            """
    ###################################################################################################
    val_loader = torch.utils.data.DataLoader(
        PriorFolder(opt.val_dir, opt.prior_dir, size=256,
                    mean=mean, std=std),
        batch_size=opt.b, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        Folder(opt.train_dir, scales=[256],
               crop=0.9, flip=True, rotate=None,
               mean=mean, std=std), collate_fn=collate_more,
        batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': 1e-4},
    ])
    logs = {'best_it':0, 'best': 0}
    sal_data_iter = iter(train_loader)
    i_sal_data = 0
    for it in tqdm(range(opt.max)):
        # if it > 1000 and it % 100 == 0:
        #     optimizer.param_groups[0]['lr'] *= 0.5
        if i_sal_data >= len(train_loader):
            sal_data_iter = iter(train_loader)
            i_sal_data = 0
        data, lbl, _ = sal_data_iter.next()
        i_sal_data += 1
        data = data.cuda()
        lbl = lbl[0].unsqueeze(1)
        noisy_label = (lbl.numpy() + np.random.binomial(1, float(p) / 100.0, (256, 256))) % 2
        noisy_label = torch.Tensor(noisy_label).cuda()
        lbl = lbl.cuda()
        msk = net(data, noisy_label)
        loss = F.binary_cross_entropy_with_logits(msk, lbl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it % 10 == 0:
            writer.add_scalar('loss', loss.item(), it)
            image = make_image_grid(data[:6], mean, std)
            writer.add_image('Image', torchvision.utils.make_grid(image), it)
            big_msk = F.sigmoid(msk).expand(-1, 3, -1, -1)
            writer.add_image('msk', torchvision.utils.make_grid(big_msk.data[:6]), it)
            big_msk = lbl.expand(-1, 3, -1, -1)
            writer.add_image('gt', torchvision.utils.make_grid(big_msk.data[:6]), it)
        # if it % 200 == 0:
        if it != 0 and it % 100 == 0:
            fm, mae = validate(val_loader, net, os.path.join(check_dir, 'results'),
                                              os.path.join(opt.val_dir, 'masks'))
            print(u'损失: %.4f'%(loss.item()))
            print(u'最大FM: iteration %d的%.4f, 这次FM: %.4f'%(logs['best_it'], logs['best'], fm))
            logs[it] = {'FM': fm}
            if fm > logs['best']:
                logs['best'] = fm
                logs['best_it'] = it
                torch.save(net.state_dict(), '%s/net-best.pth' % (check_dir))
            with open(os.path.join(check_dir, 'logs.json'), 'w') as outfile:
                json.dump(logs, outfile)
            torch.save(net.state_dict(), '%s/net-iter%d.pth' % (check_dir, it))


if __name__ == "__main__":
    main()
