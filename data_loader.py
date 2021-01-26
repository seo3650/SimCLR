import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from gaussian_blur import GaussianBlur
from torchvision import datasets

import random

class DataSetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, input_shape, dataset, color_distortion=0.8):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.input_shape = input_shape
        self.dataset = dataset
        self.color_distortion = color_distortion

    def get_data_loaders(self, option='train+unlabeled'):
        unlabel_data_augment = self._simclr_transform()
        train_data_augment = transforms.Compose([
            transforms.ToTensor()
        ])
        if self.dataset == 'STL-10': 
            if option == 'train+unlabeled':
                train_dataset = datasets.STL10('./data', split=option, download=True,
                                            transform=SimCLRDataTransform(unlabel_data_augment, option))
            elif option == 'train':
                train_dataset = datasets.STL10('./data', split=option, download=True,
                                                transform=SimCLRDataTransform(train_data_augment, option))
        elif self.dataset == 'CIFAR-10':
            assert option == 'train'
            train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                        transform=SimCLRDataTransform(unlabel_data_augment, option))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_test_data_loaders(self):
        if self.dataset == 'STL-10':
            test_dataset = datasets.STL10('./data', split='test', download=True,
                                        transform=transforms.ToTensor())
        elif self.dataset == 'CIFAR-10':
            test_dataset = datasets.CIFAR10('./data', train=False, download=True,
                                        transform=transforms.ToTensor())

        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=False)

    def _simclr_transform(self):
        # I strongly recommand you to use torchvision.transforms to implement data augmentation
        # You can use provided gaussian_blur if you want
        color_jitter = transforms.ColorJitter(brightness=self.color_distortion,
                                   contrast=self.color_distortion,
                                   saturation=self.color_distortion,
                                   hue=(-0.2, 0.2))

        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(self.input_shape[0]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size = int(0.1* self.input_shape[0])),
            transforms.ToTensor()
        ])

        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=False, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=False)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform, option):
        self.transform = transform
        self.option = option

    def __call__(self, sample):
        if self.option == 'train':
            return self.transform(sample)
        elif self.option == 'train+unlabeled':
            xi = self.transform(sample)
            xj = self.transform(sample)
            return xi, xj
        else:
            assert False
