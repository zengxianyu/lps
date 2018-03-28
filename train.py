import gc
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from model import Feature, Deconv
import model
from dataset import MyData
from datetime import datetime
import numpy as np
import os
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='/home/crow/data/datasets/saliency_Dataset/DUTS/DUT-train')  # training dataset
parser.add_argument('--val_dir', default='/home/crow/data/datasets/saliency_Dataset/DUTS/DUT-val')  # validation dataset
parser.add_argument('--check_dir', default='./parameters')  # save parameters
parser.add_argument('--m', default='conv')  # fully connected or convolutional region embedding
parser.add_argument('--e', type=int, default=36)  # epoches
parser.add_argument('--b', type=int, default=3)  # batch size
parser.add_argument('--p', type=int, default=5)  # probability of random flipping during training
opt = parser.parse_args()
print(opt)


def validation(feature, net, loader):
    total_loss = 0
    for ib, (data, lbl) in enumerate(loader):
        inputs = Variable(data).cuda()
        lbl = lbl.float()
        noisy_label = (lbl.numpy() + np.random.binomial(1, float(p)/100.0, (256, 256))) % 2
        noisy_label = Variable(torch.Tensor(noisy_label).unsqueeze(1)).cuda()
        lbl = Variable(lbl.unsqueeze(1)).cuda()

        feats = feature(inputs)
        msk = net(feats, noisy_label)

        loss = criterion(msk, lbl)
        total_loss += loss.data[0]
    return total_loss / len(loader)

# from tensorboard import SummaryWriter
# os.system('rm -rf ./runs/*')
# writer = SummaryWriter('./runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

check_root = opt.check_dir
train_data = opt.train_dir
val_data = opt.val_dir
p = opt.p
epoch = opt.e
bsize = opt.b
is_fc = False

if not os.path.exists(check_root):
    os.mkdir(check_root)

# models
feature = Feature()
feature.cuda()

constructor = 'Net_%s' % opt.m
net = getattr(model, constructor)()
net.cuda()

deconv = Deconv()
deconv.cuda()

loader = torch.utils.data.DataLoader(
            MyData(train_data, transform=True),
            batch_size=bsize*5, shuffle=True, num_workers=4, pin_memory=True)

criterion = nn.BCEWithLogitsLoss()

# optimizers
optimizer_mynet = torch.optim.Adam(net.parameters(), lr=1e-3)
optimizer_deconv = torch.optim.Adam(deconv.parameters(), lr=1e-3)
optimizer_feature = torch.optim.Adam(feature.parameters(), lr=1e-4)

for it in range(epoch):
    for ib, (data, lbl) in enumerate(loader):
        inputs = Variable(data).cuda()
        lbl = Variable(lbl.float().unsqueeze(1)).cuda()

        feats = feature(inputs)
        feats = feats[::-1]
        msk = deconv(feats[:3])
        msk = functional.upsample(msk, scale_factor=4)
        loss = criterion(msk, lbl)

        deconv.zero_grad()
        feature.zero_grad()
        loss.backward()
        optimizer_deconv.step()
        optimizer_feature.step()
        print('loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], it, ib))
        del inputs, msk, lbl, loss, feats
        gc.collect()

loader = torch.utils.data.DataLoader(
            MyData(train_data, transform=True),
            batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
            MyData(val_data, transform=True),
            batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)
min_loss = 1000.0
for it in range(epoch):
    for ib, (data, lbl) in enumerate(loader):
        inputs = Variable(data).cuda()
        lbl = lbl.float()
        noisy_label = (lbl.numpy() + np.random.binomial(1, float(p)/100.0, (256, 256))) % 2
        noisy_label = Variable(torch.Tensor(noisy_label).unsqueeze(1)).cuda()
        lbl = Variable(lbl.unsqueeze(1)).cuda()

        feats = feature(inputs)
        msk = net(feats, noisy_label)

        loss = criterion(msk, lbl)

        net.zero_grad()
        feature.zero_grad()
        loss.backward()
        optimizer_mynet.step()
        optimizer_feature.step()

        # visulize
        # image = inputs.data.cpu()
        # image[:, 0] *= 0.229
        # image[:, 1] *= 0.224
        # image[:, 2] *= 0.225
        #
        # image[:, 0] += 0.485
        # image[:, 1] += 0.456
        # image[:, 2] += 0.406
        # writer.add_image('image', torchvision.utils.make_grid(image),
        #                  ib)
        # mask1 = functional.sigmoid(msk).data.cpu().repeat(1, 3, 1, 1)
        # writer.add_image('maps', torchvision.utils.make_grid(mask1),
        #                  ib)
        # writer.add_scalar('loss', loss.data[0], ib)
        print('loss: %.4f, min-loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], min_loss, it, ib))

        del inputs, msk, lbl, loss, feats
        gc.collect()
    sb = validation(feature, net, val_loader)
    if sb < min_loss:
        filename = ('%s/net.pth' % (check_root))
        torch.save(net.state_dict(), filename)
        filename = ('%s/feature.pth' % (check_root))
        torch.save(feature.state_dict(), filename)
        print('save: (epoch: %d, step: %d)' % (it, ib))
        min_loss = sb