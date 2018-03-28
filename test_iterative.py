import torch
from torch.autograd import Variable
from torch.nn import functional
import os
from model import Feature
import model
from dataset import MyTestData
import download
import time
import PIL.Image as Image
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='/home/zeng/data/datasets/saliency_Dataset/ECSSD')
parser.add_argument('--prior_map', default='prior')  # set prior_map to the name of the directory of proir maps
parser.add_argument('--output_dir', default='/home/zeng/data/datasets/saliency_Dataset/ECSSD/pubcode')  # save checkpoint parameters
parser.add_argument('--m', default='conv')  # fully connected or convolutional region embedding
parser.add_argument('--T', type=int, default=3)  # iterations
parser.add_argument('--f', default=None)  # parameters of the feature extractor
parser.add_argument('--n', default=None)  # parameters of the network


def main():
    opt = parser.parse_args()
    print(opt)

    input_dir = opt.input_dir
    prior_map = opt.prior_map
    output_dir = opt.output_dir

    # parameters
    if opt.f is None or opt.n is None:
        param_feature = './feature_%s.pth' % opt.m
        param_net = './net_%s.pth' % opt.m
        constructor = 'download_net_%s' % opt.m
        download_net = getattr(download, constructor)
        constructor = 'download_feature_%s' % opt.m
        download_feature = getattr(download, constructor)

        if not os.path.exists(param_net):
            os.system(download_net)
        if not os.path.exists(param_feature):
            os.system(download_feature)
    else:
        param_feature = opt.f
        param_net = opt.n

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    loader = torch.utils.data.DataLoader(
        MyTestData(input_dir, transform=True, ptag=prior_map),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    feature = Feature()
    feature.cuda()
    feature.load_state_dict(torch.load(param_feature))
    feature.eval()

    constructor = 'Net_%s' % opt.m
    net = getattr(model, constructor)()
    net.cuda()
    net.load_state_dict(torch.load(param_net))
    net.eval()
    test(loader, feature, net, output_dir, opt.T)


def test(loader, feature, net, output_dir, T):
    feature.train(False)
    net.train(False)
    print('start')
    start_time = time.time()
    it = 0
    for ib, (data, msk, img_name, img_size) in enumerate(loader):
        print it

        inputs = Variable(data).cuda()

        msk = Variable(msk.unsqueeze(1)).cuda()

        feats = feature(inputs)

        for t in range(1, T + 1):
            msk = functional.sigmoid(net(feats, msk.detach())) * (1.0 / t) + msk.detach() * ((t - 1.0) / t)

        mask = msk.data[0, 0].cpu().numpy()
        mask = (mask*255).astype(np.uint8)
        mask = Image.fromarray(mask)
        mask = mask.resize((img_size[0][0], img_size[1][0]))
        mask.save(os.path.join(output_dir, img_name[0] + '.png'),'png')
        it += 1
    print('end, cost %.2f seconds for %d images' % (time.time() - start_time, it))


if __name__ == '__main__':
    main()
