import os
import numpy as np
import PIL.Image as Image
import torch
from torch.utils import data
import pdb
import random


class ImageFiles(data.Dataset):
    def __init__(self, img_dir, prior_dir,
                 size = 256,
                 mean=None, std=None):
        super(ImageFiles, self).__init__()
        self.mean, self.std = mean, std
        self.size = size
        names = os.listdir(img_dir)
        names = ['.'.join(name.split('.')[:-1]) for name in names]
        self.img_filenames = list(map(lambda x: os.path.join(img_dir, x+'.jpg'), names))
        self.pr_filenames = list(map(lambda x: os.path.join(prior_dir, x+'.png'), names))
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_filenames[index]
        img = Image.open(img_file)
        pr_file = self.pr_filenames[index]
        pr = Image.open(pr_file)
        name = self.names[index]
        WW, HH = img.size
        img = img.resize((self.size, self.size))
        img = np.array(img, dtype=np.float64)/255
        pr = pr.resize((self.size, self.size))
        pr = np.array(pr, dtype=np.float64)/255
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        pr = torch.from_numpy(pr).float()
        return img, pr, name, WW, HH


if __name__ == "__main__":
    sb = ImageFiles('../../data/datasets/ILSVRC14VOC/images')
    pdb.set_trace()
