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


class FCN(nn.Module):
    def __init__(self, net):
        super(FCN, self).__init__()
        if 'vgg' in net.base:
            dims = dim_dict['vgg'][::-1]
        else:
            dims = dim_dict[net.base][::-1]
        self.preds = nn.ModuleList([nn.Conv2d(d, 1, kernel_size=1) for d in dims])
        self.upscales = nn.ModuleList([nn.ConvTranspose2d(1, 1, 1, 1, 0)]*2+[nn.ConvTranspose2d(1, 1, 4, 2, 1)]*2)
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
        self.feature = net.feature
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad=False

    def forward(self, *data):
        x = data[0]
        self.feature.feats[x.device.index] = []
        x = self.feature(x)
        feats = self.feature.feats[x.device.index]
        feats += [x]
        feats = feats[::-1]
        pred = self.preds[0](feats[0])
        preds = [pred]
        for i in range(4):
            pred = self.preds[i+1](feats[i+1]+self.upscales[i](pred))
            preds += [pred]
        if self.training:
            return preds
        else:
            return pred



if __name__ == "__main__":
    net = Net()
    net.cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    z = torch.Tensor(2, 1, 256, 256).cuda()
    sb = net(x, z)
    pdb.set_trace()
