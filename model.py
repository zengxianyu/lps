import numpy as np
from myfunc import get_upsample_filter
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision
from torch.autograd.variable import Variable
import pdb
from functools import reduce
from torch.nn import init


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.features = \
        [nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.MaxPool2d(1, stride=1, ceil_mode=True),  # 1/8
            # conv4
            nn.Conv2d(256, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
        ),
        nn.Sequential(
            nn.MaxPool2d(1, stride=1, ceil_mode=True),  # 1/16
            # conv5 features
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=4, dilation=4),
            nn.ReLU(),
        )]
        self.features = nn.ModuleList(self.features)
        vgg16 = torchvision.models.vgg16(pretrained=True)
        for l1, l2 in zip(list(vgg16.features),
                          reduce(lambda x, y: list(x) + list(y), self.features)):
            if (isinstance(l1, nn.Conv2d) and
                    isinstance(l2, nn.Conv2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
            if (isinstance(l1, nn.BatchNorm2d) and
                    isinstance(l2, nn.BatchNorm2d)):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

    def forward(self, x):
        feats = []
        for f in self.features:
            x = f(x)
            feats.append(x)
        return feats


# fully connected region embedding
class Net_fc(nn.Module):
    def __init__(self):
        super(Net_fc, self).__init__()
        self.v = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=1),
                    nn.BatchNorm2d(64)
                ),
                nn.Sequential(
                    nn.Conv2d(128, 64 * 4, kernel_size=1),
                    nn.BatchNorm2d(64 * 4),
                    nn.PixelShuffle(2)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 64 * 16, kernel_size=1),
                    nn.BatchNorm2d(64 * 16),
                    nn.PixelShuffle(4)
                ),
                nn.Sequential(
                    nn.Conv2d(512, 64 * 16, kernel_size=1),
                    nn.BatchNorm2d(64 * 16),
                    nn.PixelShuffle(4)
                ),
                nn.Sequential(
                    nn.Conv2d(512, 64 * 16, kernel_size=1),
                    nn.BatchNorm2d(64 * 16),
                    nn.PixelShuffle(4)
                )
            ]
        )
        self.u = nn.Conv2d(64 * 5, 512, kernel_size=1)
        self.w = nn.Linear(64*5, 512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
        for i in range(1, 5):
            init.xavier_normal(self.v[i][0].weight.data[0])
            self.v[i][0].weight.data[1:].copy_(self.v[i][0].weight.data[0:1])
            m.bias.data.fill_(0)

    def forward(self, feats, map):
        fs = []
        for i in range(5):
            fs.append(self.v[i](feats[i]))
        feats = torch.cat(fs, 1)

        feats_c1 = feats * map.expand_as(feats)
        feats_c2 = feats * (1-map).expand_as(feats)
        c1 = feats_c1.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False) / map.sum()
        c2 = feats_c2.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False) / (1-map).sum()
        c1 = self.w(c1).unsqueeze(2).unsqueeze(3)
        c2 = self.w(c2).unsqueeze(2).unsqueeze(3)

        feats = self.u(feats)

        dist1 = (feats - c1.expand_as(feats))**2
        dist1 = torch.sqrt(dist1.sum(dim=1, keepdim=True))
        dist2 = (feats - c2.expand_as(feats)) ** 2
        dist2 = torch.sqrt(dist2.sum(dim=1, keepdim=True))

        return dist2 - dist1


# convolutional region embedding
class Net_conv(nn.Module):
    def __init__(self):
        super(Net_conv, self).__init__()
        self.v = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=1),
                    nn.BatchNorm2d(64)
                ),
                nn.Sequential(
                    nn.Conv2d(128, 64 * 4, kernel_size=1),
                    nn.BatchNorm2d(64 * 4),
                    nn.PixelShuffle(2)
                ),
                nn.Sequential(
                    nn.Conv2d(256, 64 * 16, kernel_size=1),
                    nn.BatchNorm2d(64 * 16),
                    nn.PixelShuffle(4)
                ),
                nn.Sequential(
                    nn.Conv2d(512, 64 * 16, kernel_size=1),
                    nn.BatchNorm2d(64 * 16),
                    nn.PixelShuffle(4)
                ),
                nn.Sequential(
                    nn.Conv2d(512, 64 * 16, kernel_size=1),
                    nn.BatchNorm2d(64 * 16),
                    nn.PixelShuffle(4)
                )
            ]
        )
        self.u = nn.Conv2d(64 * 5 * 2, 512, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
        for i in range(1, 5):
            init.xavier_normal(self.v[i][0].weight.data[0])
            self.v[i][0].weight.data[1:].copy_(self.v[i][0].weight.data[0:1])
            m.bias.data.fill_(0)

    def forward(self, feats, map):
        fs = []
        for i in range(5):
            fs.append(self.v[i]( feats[i] ))
        feats = torch.cat(fs, 1)

        feats = self.u(
            torch.cat((feats, feats), 1)
        )
        feats_c1 = feats * map.expand_as(feats)
        feats_c2 = feats * (1-map).expand_as(feats)
        c1 = feats_c1.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) / map.sum()
        c2 = feats_c2.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) / (1-map).sum()

        dist1 = (feats - c1.expand_as(feats))**2
        dist1 = torch.sqrt(dist1.sum(dim=1, keepdim=True))
        dist2 = (feats - c2.expand_as(feats)) ** 2
        dist2 = torch.sqrt(dist2.sum(dim=1, keepdim=True))

        return dist2 - dist1


class Deconv(nn.Module):
    def __init__(self):
        super(Deconv, self).__init__()
        _reduce_dimension = [
            nn.Conv2d(512, 32, 1),
            nn.Conv2d(512, 32, 1),
            nn.Conv2d(256, 32, 1)
        ]
        _prediction = [
            nn.Conv2d(32, 1, 1),
            nn.Conv2d(33, 1, 1),
            nn.Conv2d(33, 1, 1),
        ]
        self.reduce_dimension = nn.ModuleList(_reduce_dimension)
        self.prediction = nn.ModuleList(_prediction)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, feats):
        for i in range(len(feats)):
            feats[i] = self.reduce_dimension[i](feats[i])
        y = self.prediction[0](feats[0])
        for i in range(1, len(feats)):
            y = self.prediction[i](
                torch.cat((feats[i], y), 1)
            )
        return y


