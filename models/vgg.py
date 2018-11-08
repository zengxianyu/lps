import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
from torch.autograd.variable import Variable
import pdb


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for f in self.features:
            x = f(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    list_feature = list(model.features)
    _features = [nn.Sequential(*list_feature[:3]),
                 nn.Sequential(*list_feature[3:6]),
                 nn.Sequential(*list_feature[6:11]),
                 nn.Sequential(*list_feature[11:16]),
                 nn.Sequential(*list_feature[16:21])]
    model.features = nn.ModuleList(_features)
    model.classifier = None
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    list_feature = list(model.features)
    _features = [nn.Sequential(*list_feature[:4]),
                 nn.Sequential(*list_feature[4:8]),
                 nn.Sequential(*list_feature[8:15]),
                 nn.Sequential(*list_feature[15:22]),
                 nn.Sequential(*list_feature[22:29])]
    model.features = nn.ModuleList(_features)
    model.classifier = None
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    list_feature = list(model.features)
    _features = [nn.Sequential(*list_feature[:5]),
                 nn.Sequential(*list_feature[5:10]),
                 nn.Sequential(*list_feature[10:15]),
                 nn.Sequential(*list_feature[15:20]),
                 nn.Sequential(*list_feature[20:25])]
    model.features = nn.ModuleList(_features)
    model.classifier = None
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    list_feature = list(model.features)
    _features = [nn.Sequential(*list_feature[:7]),
                 nn.Sequential(*list_feature[7:14]),
                 nn.Sequential(*list_feature[14:21]),
                 nn.Sequential(*list_feature[21:28]),
                 nn.Sequential(*list_feature[28:35])]
    model.features = nn.ModuleList(_features)
    model.classifier = None
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
        # model.load_state_dict(torch.load('/home/crow/SPN.pytorch/demo/models/vgg16_from_caffe.pth'))
    list_feature = list(model.features)
    _features = [nn.Sequential(*list_feature[:5]),
                 nn.Sequential(*list_feature[5:10]),
                 nn.Sequential(*list_feature[10:17]),
                 nn.Sequential(*list_feature[17:24]),
                 nn.Sequential(*list_feature[24:31])]
    model.features = nn.ModuleList(_features)
    model.classifier = None
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    list_feature = list(model.features)
    _features = [nn.Sequential(*list_feature[:7]),
                 nn.Sequential(*list_feature[7:14]),
                 nn.Sequential(*list_feature[14:24]),
                 nn.Sequential(*list_feature[24:34]),
                 nn.Sequential(*list_feature[34:44])]
    model.features = nn.ModuleList(_features)
    model.classifier = None
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    list_feature = list(model.features)
    _features = [nn.Sequential(*list_feature[:5]),
                 nn.Sequential(*list_feature[5:10]),
                 nn.Sequential(*list_feature[10:19]),
                 nn.Sequential(*list_feature[19:28]),
                 nn.Sequential(*list_feature[28:37])]
    model.features = nn.ModuleList(_features)
    model.classifier = None
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    list_feature = list(model.features)
    _features = [nn.Sequential(*list_feature[:7]),
                 nn.Sequential(*list_feature[7:14]),
                 nn.Sequential(*list_feature[14:27]),
                 nn.Sequential(*list_feature[27:40]),
                 nn.Sequential(*list_feature[40:53])]
    model.features = nn.ModuleList(_features)
    model.classifier = None
    return model


if __name__ == "__main__":
    vgg = vgg16(pretrained=True).cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    sb = vgg(Variable(x))
    pdb.set_trace()