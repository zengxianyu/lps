from dataset import MyTestData, MyData
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os.path as osp
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from skimage import color, img_as_ubyte


def tensor2image(image):
    """
    convert a mean-0 tensor to float numpy image
    :param image:
    :return: image
    """
    image = image.clone()
    image[0] += 122.67891434 / 255
    image[1] += 116.66876762 / 255
    image[2] += 104.00698793 / 255
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    image = img_as_ubyte(image)
    return image


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


def get_loader_thus(opt, ptag=None):
    data_root = opt.data
    if opt.phase == 'train':
        loader = torch.utils.data.DataLoader(
            MyData(data_root, transform=True),
            batch_size=opt.bsize, shuffle=True, num_workers=4, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(
            MyTestData(data_root, transform=True, ptag=ptag),
            batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    return loader


def pr_plot(dataset, first):
    mat_root = '/home/zeng/data/datasets/saliency_Dataset/mat'
    tags = [first, 'Ours', 'SRM', 'MCDL', 'ELD', 'RFCN']
    fig = plt.figure()
    for tag in tags:
        mat_data = loadmat(osp.join(mat_root, dataset, tag+'.mat'))
        print 'dataset: %s, alg: %s' %(dataset, tag)
        print mat_data['mFmeasure']
        if tag == first:
            plt.plot(mat_data['mRecall'], mat_data['mPre'], linewidth=4, label=tag)
        elif tag == 'Ours_mnet':
            plt.plot(mat_data['mRecall'], mat_data['mPre'], linewidth=4, label=tag)
        else:
            plt.plot(mat_data['mRecall'], mat_data['mPre'], linewidth=1, label=tag)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xlabel('Recall', fontsize=16)
    plt.ylabel('Precision', fontsize=16)
    fig.savefig(dataset+'_pr.pdf')

    fig = plt.figure()
    for tag in tags:
        mat_data = loadmat(osp.join(mat_root, dataset, tag+'.mat'))
        fm = 1.3 * (mat_data['mPre'] * mat_data['mRecall'] / (0.3 * mat_data['mPre'] + mat_data['mRecall']))
        if tag == first:
            plt.plot(np.linspace(0, 1, 21), fm[::-1], linewidth=4, label=tag)
        elif tag == 'Ours_mnet':
            plt.plot(np.linspace(0, 1, 21), fm[::-1], linewidth=4, label=tag)
        else:
            plt.plot(np.linspace(0, 1, 21), fm[::-1], linewidth=1, label=tag)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xlabel('Threshold', fontsize=16)
    plt.ylabel('F-Score', fontsize=16)
    fig.savefig(dataset+'_fm.pdf')

