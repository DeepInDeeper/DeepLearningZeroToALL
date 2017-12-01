#coding:utf8
from __future__ import print_function
from __future__ import division
import numpy as np 
import pandas as pd 

import numpy as np
import random
from datetime import datetime
from sklearn.utils import shuffle

import torch
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from models import XnumpyToTensor,YnumpyToTensor

use_cuda = torch.cuda.is_available()
BASE_FOLDER = u'/media/yijie/文档/dataset/kaggle_Iceberg'

def fixSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

class IcebergCustomDataSet(Dataset):
    """total dataset."""

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'image': self.data[idx, :, :, :], 'labels': np.asarray([self.labels[idx]])}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = image.astype(float) / 255
        return {'image': torch.from_numpy(image.copy()).float(),
                'labels': torch.from_numpy(labels).float()
                }


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        image, labels = sample['image'], sample['labels']

        if random.random() < 0.5:
            image = np.flip(image, 1)

        return {'image': image, 'labels': labels}


class RandomVerticallFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if random.random() < 0.3:
            image = np.flip(image, 0)
        return {'image': image, 'labels': labels}


class RandomTranspose(object):
    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']
        if random.random() < 0.7:
            image = np.transpose(image, 0)
        return {'image': image, 'labels': labels}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        img = tensor['image'].float()
        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'image': img, 'labels': tensor['labels']}

class FullTrainningDataset(torch.utils.data.Dataset):
    def __init__(self, full_ds, offset, length):
        self.full_ds = full_ds
        self.offset = offset
        self.length = length
        assert len(full_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(FullTrainningDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i + self.offset]

def trainTestSplit(dataset, val_share=0.11):
    val_offset = int(len(dataset) * (1 - val_share))
    # print("Offest:" + str(val_offset))
    return FullTrainningDataset(dataset, 0, val_offset), FullTrainningDataset(dataset,val_offset, len(dataset) - val_offset)





# def readSuffleData(seed=datetime.now()):
def readSuffleData(seed_num):
    fixSeed(seed_num)
    local_data = pd.read_json(BASE_FOLDER + '/train.json')

    local_data = shuffle(local_data)  # otherwise same validation set each time!
    local_data = local_data.reindex(np.random.permutation(local_data.index))
    # local_data = shuffle(local_data)  # otherwise same validation set each time!
    # local_data = local_data.reindex(np.random.permutation(local_data.index))
    local_data['band_1'] = local_data['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
    local_data['band_2'] = local_data['band_2'].apply(lambda x: np.array(x).reshape(75, 75))
    local_data['inc_angle'] = pd.to_numeric(local_data['inc_angle'], errors='coerce')
    band_1 = np.concatenate([im for im in local_data['band_1']]).reshape(-1, 75, 75)
    band_2 = np.concatenate([im for im in local_data['band_2']]).reshape(-1, 75, 75)
    local_full_img = np.stack([band_1, band_2], axis=1)
    return local_data, local_full_img



def getTrainValLoaders(data,full_img,batch_size,num_workers):
    # global train_ds, val_ds, train_loader, val_loader
    train_imgs = XnumpyToTensor(full_img)
    train_targets = YnumpyToTensor(data['is_iceberg'].values)
    dset_train = TensorDataset(train_imgs, train_targets)
    local_train_ds, local_val_ds = trainTestSplit(dset_train)
    local_train_loader = torch.utils.data.DataLoader(local_train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    local_val_loader = torch.utils.data.DataLoader(local_val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return local_train_loader, local_val_loader, local_train_ds, local_val_ds


def getCustomTrainValLoaders():
    # global train_ds, train_loader, val_loader
    from random import randrange
    X_train, X_val, y_train, y_val = train_test_split(full_img, data['is_iceberg'].values,
                                                      test_size=validationRatio,
                                                      random_state=global_seed)
    local_train_ds = IcebergCustomDataSet(X_train, y_train,
                                          transform=transforms.Compose([

                                            RandomHorizontalFlip(),
                                            RandomVerticallFlip(),
                                            ToTensor(),
                                            # Normalize(mean = [0.456],std =[0.229]),
                                        ]))
    local_val_ds = IcebergCustomDataSet(X_val, y_val,
                                       transform=transforms.Compose([
                                           RandomHorizontalFlip(),
                                           RandomVerticallFlip(),
                                           ToTensor(),
                                           # Normalize(mean=[0.456], std=[0.229]),
                                       ]))
    local_train_loader = DataLoader(dataset=local_train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    local_val_loader = DataLoader(dataset=local_val_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    # print(local_train_loader)
    # print(local_val_loader)
    return local_train_loader, local_val_loader, local_train_ds, local_val_ds
