import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd.variable import Variable

# from .densenet import *
# from .resnet import *
from .vgg import *

# from densenet import *
# from resnet import *
# from vgg import *

import numpy as np
import sys
thismodule = sys.modules[__name__]
import pdb

img_size = 256

dim_dict = {
    'vgg': [64, 128, 256, 512, 512]
}


def proc_vgg(model):
    def hook(module, input, output):
        model.feats[output.device.index] += [output]
    for m in model.features[:-1]:
        m[-2].register_forward_hook(hook)
    # dilation
    def remove_sequential(all_layers, network):
        for layer in network.children():
            if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
                remove_sequential(all_layers, layer)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)
    model.features[2][-1].stride = 1
    model.features[2][-1].kernel_size = 1
    all_layers = []
    remove_sequential(all_layers, model.features[3])
    for m in all_layers:
        if isinstance(m, nn.Conv2d):
            m.dilation = (2, 2)
            m.padding = (2, 2)

    model.features[3][-1].stride = 1
    model.features[3][-1].kernel_size = 1
    all_layers = []
    remove_sequential(all_layers, model.features[4])
    for m in model.features[4]:
        if isinstance(m, nn.Conv2d):
            m.dilation = (4, 4)
            m.padding = (4, 4)
    model.features[4][-1].stride = 1
    model.features[4][-1].kernel_size = 1
    return model


procs = {
    'vgg16': proc_vgg,
    }


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class Net(nn.Module):
    def __init__(self, pretrained=True, base='vgg16'):
        super(Net, self).__init__()
        if 'vgg' in base:
            dims = dim_dict['vgg'][::-1]
        else:
            dims = dim_dict[base][::-1]
        self.base = base
        odims = [64]*5
        hdim = 512
        self.classifier = nn.Linear(512, 1)
        self.proc_feats_list = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(dims[0], dims[0], 8, 4, 2), nn.Conv2d(dims[0], odims[0], kernel_size=3, padding=1)),
            nn.Sequential(nn.ConvTranspose2d(dims[1], dims[1], 8, 4, 2), nn.Conv2d(dims[1], odims[1], kernel_size=3, padding=1)),
            nn.Sequential(nn.ConvTranspose2d(dims[2], dims[2], 8, 4, 2), nn.Conv2d(dims[2], odims[2], kernel_size=3, padding=1)),
            nn.Sequential(nn.ConvTranspose2d(dims[3], dims[3], 4, 2, 1), nn.Conv2d(dims[3], odims[3], kernel_size=3, padding=1)),
            # nn.Sequential(nn.Conv2d(dims[0], odims[0]*16, kernel_size=3, padding=1), nn.PixelShuffle(4)),
            # nn.Sequential(nn.Conv2d(dims[1], odims[1]*16, kernel_size=3, padding=1), nn.PixelShuffle(4)),
            # nn.Sequential(nn.Conv2d(dims[2], odims[2]*16, kernel_size=3, padding=1), nn.PixelShuffle(4)),
            # nn.Sequential(nn.Conv2d(dims[3], odims[3]*4, kernel_size=3, padding=1), nn.PixelShuffle(2)),
            nn.Conv2d(dims[4], dims[4], kernel_size=3, padding=1),
        ])
        self.proc_feats = nn.Conv2d(sum(odims), hdim, kernel_size=3, padding=1)
        self.proc_mul = nn.Conv2d(sum(odims), hdim, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
        self.feature = getattr(thismodule, base)(pretrained=pretrained)
        self.feature.feats = {}
        self.feature = procs[base](self.feature)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad=False

    def forward(self, x, prior):
        self.feature.feats[x.device.index] = []
        x = self.feature(x)
        feats = self.feature.feats[x.device.index]
        feats += [x]
        feats = feats[::-1]
        for i, p in enumerate(self.proc_feats_list): feats[i]=p(feats[i])
        feats = torch.cat(feats, 1)
        c1 = self.proc_mul(feats*prior).sum(3, keepdim=True).sum(2, keepdim=True)/(prior.sum())
        c2 = self.proc_mul(feats*(1-prior)).sum(3, keepdim=True).sum(2, keepdim=True)/((1-prior).sum())
        feats = self.proc_feats(feats)
        dist1 = (feats - c1) ** 2
        dist1 = torch.sqrt(dist1.sum(dim=1, keepdim=True))
        dist2 = (feats - c2) ** 2
        dist2 = torch.sqrt(dist2.sum(dim=1, keepdim=True))
        return dist2 - dist1



if __name__ == "__main__":
    net = Net()
    net.cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    z = torch.Tensor(2, 1, 256, 256).cuda()
    sb = net(x, z)
    pdb.set_trace()
