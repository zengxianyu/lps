import torch
from torch.autograd import Variable
from torch.nn import functional
import os
import cv2
import matplotlib.pyplot as plt
from model import Feature, Net_conv, Net_fc
from dataset import MyTestData
from download import *

# use fully connected region embedding
is_fc = False

# the number of iterations
T = 3

# valid_data_dir contains two subdirectory: a directory of input images named "Images",
# and a directory of prior maps.
valid_data_dir = '/home/zeng/data/datasets/saliency_Dataset/ECSSD'

# set prior_map to the name of the directory of proir maps
prior_map = 'SRM'

output_dir = '/home/zeng/data/datasets/saliency_Dataset/ECSSD/pubcode'

# parameters
if is_fc:
    param_feature = './feature_fc.pth'
    param_net = './net_fc.pth'
    download_net = download_net_fc
    download_feature = download_feature_fc
else:
    param_feature = './feature_conv.pth'
    param_net = './net_conv.pth'
    download_net = download_net_conv
    download_feature = download_feature_conv

if not os.path.exists(param_net):
    os.system(download_net)
if not os.path.exists(param_feature):
    os.system(download_feature)


if not os.path.exists(output_dir):
    os.mkdir(output_dir)

loader = torch.utils.data.DataLoader(
            MyTestData(valid_data_dir, transform=True, ptag=prior_map),
            batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

feature = Feature()
feature.cuda()
feature.load_state_dict(torch.load(param_feature))

if is_fc:
    net = Net_fc()
else:
    net = Net_conv()
net.cuda()
net.load_state_dict(torch.load(param_net))

for ib, (data, msk, img_name, img_size) in enumerate(loader):
    print ib
    inputs = Variable(data).cuda()

    msk = Variable(msk.unsqueeze(1)).cuda()
    feats = feature(inputs)
    for t in range(1, T+1):
        msk = functional.sigmoid(net(feats, msk.detach())) * (1.0 / t) + msk.detach() * ((t - 1.0) / t)
    mask = msk.data[0, 0].cpu().numpy()
    mask = cv2.resize(mask, dsize=(img_size[0][0], img_size[1][0]))
    plt.imsave(os.path.join(output_dir, img_name[0] + '.png'), mask, cmap='gray')
