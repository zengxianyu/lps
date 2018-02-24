import gc
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from model import Feature, Net_conv, Net_fc, Deconv
from dataset import MyData
from datetime import datetime
import numpy as np
import os
# from tensorboard import SummaryWriter
# os.system('rm -rf ./runs/*')
# writer = SummaryWriter('./runs/'+datetime.now().strftime('%B%d  %H:%M:%S'))

check_root = './parameters'
# train_data contains two subdirectory: Images (training images) and Masks (ground truth maps)
train_data = '/home/crow/data/datasets/saliency_Dataset/ADUTS'
p = 5
epoch = 3
bsize = 6
is_fc = False

if not os.path.exists(check_root):
    os.mkdir(check_root)

# models
feature = Feature()
feature.cuda()
# feature.load_state_dict(torch.load('/home/zeng/data/models/torch/haha/fcn/feature-epoch-13-step-4000.pth'))

if is_fc:
    net = Net_fc()
else:
    net = Net_conv()
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
        lbl = Variable(lbl.unsqueeze(1)).cuda()

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
for it in range(epoch):
    for ib, (data, lbl) in enumerate(loader):
        inputs = Variable(data).cuda()
        noisy_label = (lbl.cpu().numpy() + np.random.binomial(1, float(p)/100.0, (256, 256))) % 2
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
        print('loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], it, ib))

        del inputs, msk, lbl, loss, feats
        gc.collect()
        if ib % 10000 == 0:
            filename = ('%s/mynet-epoch-%d-step-%d.pth' % (check_root, it, ib))
            torch.save(net.state_dict(), filename)
            filename = ('%s/feature-epoch-%d-step-%d.pth' % (check_root, it, ib))
            torch.save(feature.state_dict(), filename)
            print('save: (epoch: %d, step: %d)' % (it, ib))