import sys
from matplotlib.style import use
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from PIL import Image
  


import torchvision.transforms.functional as transF
import random

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


# class TSData(Dataset):
#     def __init__(self, root_dir, Training=True, transform=None, use_mean=None, use_median=None, use_mode=None, use_50=None):
#         self.train = Training
#         self.root_dir = root_dir
#         self.transform = transform
#         self.use_mean = use_mean
#         self.use_median = use_median
#         self.use_mode = use_mode
#         self.use_50 = use_50
        
#     def __len__(self):

#         count = 0
#         for fn in os.listdir(self.root_dir):
#             count = count + 1
#         return count

#     def __getitem__(self, idx):

#         dir_idx = idx + 1
#         mean_name = str(dir_idx) + r'/mean.npy'
#         median_name = str(dir_idx) + r'/median.npy'
#         mode_name = str(dir_idx) + r'/mode.npy'
#         video_log_name = str(dir_idx) + r'/video_log.txt'
#         Flag_name = str(dir_idx) + r'/Flag.txt'

#         if not self.use_50:


        

#         data = np.load(os.path.join(self.root_dir, data_name))[0:28, :]
#         #data = np.load(os.path.join(self.root_dir, data_name))[0:68, :]
#         #data = np.load(os.path.join(self.root_dir, data_name))
#         video_log = str(np.loadtxt(os.path.join(self.root_dir, video_log_name), dtype=np.str))
#         # video_log = ''
#         Flag = np.loadtxt(os.path.join(self.root_dir, Flag_name), dtype=np.str)

#         if Flag == 'Real':
#             label = 1
#         else:
#             label = 0

#         if self.transform:
#             data = self.transform(data).squeeze()

#         return (data, label, video_log, idx)

class TSData(Dataset):
    def __init__(self, root_dir, Training=True, transform=None, use_mean=None, use_median=None, use_mode=None, use_50=None):
        self.train = Training
        self.root_dir = root_dir
        self.transform = transform
        self.use_mean = use_mean
        self.use_median = use_median
        self.use_mode = use_mode
        self.use_50 = use_50
        
    def __len__(self):

        count = 0
        for fn in os.listdir(self.root_dir):
            count = count + 1
        return count

    def __getitem__(self, idx):

        dir_idx = idx + 1
        # mean_name = str(dir_idx) + r'/mean.npy'
        # median_name = str(dir_idx) + r'/median.npy'
        # mode_name = str(dir_idx) + r'/mode.npy'
        data_name = str(dir_idx) + r'/data.npy'
        video_log_name = str(dir_idx) + r'/video_log.txt'
        Flag_name = str(dir_idx) + r'/Flag.txt'
        #使用新data则下面if语句不用
        #######
        # if not self.use_50:
        #     mean = np.load(os.path.join(self.root_dir, mean_name))[0:14, :]
        #     median = np.load(os.path.join(self.root_dir, median_name))[0:14, :]
        #     mode = np.load(os.path.join(self.root_dir, mode_name))[0:14, :]
        # else:
        #     mean = np.load(os.path.join(self.root_dir, mean_name))[14:28, :]
        #     median = np.load(os.path.join(self.root_dir, median_name))[14:28, :]
        #     mode = np.load(os.path.join(self.root_dir, mode_name))[14:28, :]
        #######
        #使用新data则下面重整data数据不用
        #######
        # data = np.zeros(((self.use_mean + self.use_median + self.use_mode)*14, 64))
        # tmp = 0
        # if self.use_mean:
        #     data[tmp:tmp+14, :] = mean
        #     tmp += 14
        # if self.use_median:
        #     data[tmp:tmp+14, :] = median
        #     tmp += 14
        # if self.use_mode:
        #     data[tmp:tmp+14, :] = mode
        #######
        #新data最终读取
        data = np.load(os.path.join(self.root_dir, data_name))
        video_log = str(np.loadtxt(os.path.join(self.root_dir, video_log_name), dtype=np.str))
        Flag = np.loadtxt(os.path.join(self.root_dir, Flag_name), dtype=np.str)

        if Flag == 'Real':
            label = 1
        else:
            label = 0

        if self.transform:
            data = self.transform(data).squeeze()

        return (data, label, video_log, idx)


class TSData_test(Dataset):
    def __init__(self, root_dir, Training=True, transform=None, feature_num=49):
        self.train = Training
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):

        count = 0
        for fn in os.listdir(self.root_dir):
            count = count + 1
        return count

    def __getitem__(self, idx):

        dir_idx = idx + 1
        #1、改测试集data存储或者2、改这里data读取（改1优先（完成））
        data_name = str(dir_idx) + r'/data.npy'
        video_log_name = str(dir_idx) + r'/video_log.txt'
        Flag_name = str(dir_idx) + r'/Flag.txt'

        data = np.load(os.path.join(self.root_dir, data_name))
        video_log = str(np.loadtxt(os.path.join(self.root_dir, video_log_name), dtype=np.str))
        Flag = np.loadtxt(os.path.join(self.root_dir, Flag_name), dtype=np.str)

        if Flag == 'Real':
            label = 1
        else:
            label = 0

        if self.transform:
            data = self.transform(data).squeeze()

        return (data, label, self.ori_graph_edges, video_log, idx)
