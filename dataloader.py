import glob
import os
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageEnhance

random.seed(1143)

def populate_train_list(lowlight_images_paths):
    train_list = []

    for lowlight_images_path in lowlight_images_paths.split(','):
        lowlight_images_path = lowlight_images_path.strip()
        image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg") + glob.glob(lowlight_images_path + "*.png")

        train_list += image_list_lowlight

    random.shuffle(train_list)

    return train_list


class lowlight_loader(data.Dataset):

    def __init__(self, lowlight_images_path, size=256, preload=False):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = size
        self.preload = preload

        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))
        if self.preload:
            self.data_list = []
            print('preloading to memory')
            for data_lowlight_path in self.train_list:
                data_lowlight = Image.open(data_lowlight_path)

                data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)

                data_lowlight = (np.asarray(data_lowlight) / 255.0)
                data_lowlight = torch.from_numpy(data_lowlight).float()
                self.data_list.append(data_lowlight.permute(2, 0, 1))

    def __getitem__(self, index):
        if self.preload:
            return self.data_list[index]
        else:
            data_lowlight_path = self.data_list[index]

            data_lowlight = Image.open(data_lowlight_path)

            data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)

            data_lowlight = (np.asarray(data_lowlight) / 255.0)
            data_lowlight = torch.from_numpy(data_lowlight).float()

            return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

class lowlight_loader_con(data.Dataset):

    def __init__(self, lowlight_images_path, size=256, preload=False):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = size
        self.preload = preload

        self.data_list = self.train_list
        self.con_list = self.train_list
        print("Total training examples:", len(self.train_list))
        if self.preload:
            self.data_list = []
            self.con_list = []
            print('preloading to memory')
            for data_lowlight_path in self.train_list:
                data_lowlight = Image.open(data_lowlight_path)

                data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)
                low_con_img = ImageEnhance.Contrast(data_lowlight).enhance(0.5)

                data_lowlight = (np.asarray(data_lowlight) / 255.0)
                data_lowlight = torch.from_numpy(data_lowlight).float()
                self.data_list.append(data_lowlight.permute(2, 0, 1))

                low_con_img = (np.asarray(low_con_img) / 255.0)
                low_con_img = torch.from_numpy(low_con_img).float()
                self.con_list.append(low_con_img.permute(2, 0, 1))

    def __getitem__(self, index):
        if self.preload:
            return self.data_list[index], self.con_list[index]
        else:
            data_lowlight_path = self.data_list[index]

            data_lowlight = Image.open(data_lowlight_path)

            data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)

            data_lowlight = (np.asarray(data_lowlight) / 255.0)
            data_lowlight = torch.from_numpy(data_lowlight).float()

            return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

class lowlight_loader_supervised(data.Dataset):

    def __init__(self, lowlight_images_path, size=256, preload=False):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = size
        self.preload = preload

        self.data_list = self.train_list
        self.gt_list = [os.path.join(lowlight_images_path, 'gt', os.path.basename(x)) for x in self.train_list]
        print("Total training examples:", len(self.train_list))
        if self.preload:
            self.data_list = []
            temp_gt = []
            print('preloading to memory')
            for data_lowlight_path, data_gt_path in zip(self.train_list, self.gt_list):
                data_lowlight = Image.open(data_lowlight_path)

                data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)

                data_lowlight = (np.asarray(data_lowlight) / 255.0)
                data_lowlight = torch.from_numpy(data_lowlight).float()
                self.data_list.append(data_lowlight.permute(2, 0, 1))

                data_gt = Image.open(data_gt_path)

                data_gt = data_gt.resize((self.size, self.size), Image.ANTIALIAS)

                data_gt = (np.asarray(data_gt) / 255.0)
                data_gt = torch.from_numpy(data_gt).float()
                temp_gt.append(data_gt.permute(2, 0, 1))

            self.gt_list = temp_gt

    def __getitem__(self, index):
        if self.preload:
            return self.data_list[index], self.gt_list[index]
        else:
            data_lowlight_path = self.data_list[index]

            data_lowlight = Image.open(data_lowlight_path)

            data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)

            data_lowlight = (np.asarray(data_lowlight) / 255.0)
            data_lowlight = torch.from_numpy(data_lowlight).float()

            return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

class lowlight_loader_supervised_sid(data.Dataset):

    def __init__(self, lowlight_images_path, size=256, preload=False):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = size
        self.preload = preload

        self.data_list = self.train_list
        self.gt_list = [os.path.join(lowlight_images_path, 'gt', self.get_long_image(x)) for x in self.train_list]
        print("Total training examples:", len(self.train_list))
        if self.preload:
            self.data_list = []
            temp_gt = []
            print('preloading to memory')
            for data_lowlight_path, data_gt_path in zip(self.train_list, self.gt_list):
                data_lowlight = Image.open(data_lowlight_path)

                data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)

                data_lowlight = (np.asarray(data_lowlight) / 255.0)
                data_lowlight = torch.from_numpy(data_lowlight).float()
                self.data_list.append(data_lowlight.permute(2, 0, 1))

                data_gt = Image.open(data_gt_path)

                data_gt = data_gt.resize((self.size, self.size), Image.ANTIALIAS)

                data_gt = (np.asarray(data_gt) / 255.0)
                data_gt = torch.from_numpy(data_gt).float()
                temp_gt.append(data_gt.permute(2, 0, 1))

            self.gt_list = temp_gt

    def get_long_image(self, x):
        img_id = os.path.basename(x).split('_')[0]
        return img_id + '_00_10s.jpg'

    def __getitem__(self, index):
        if self.preload:
            return self.data_list[index], self.gt_list[index]
        else:
            data_lowlight_path = self.data_list[index]

            data_lowlight = Image.open(data_lowlight_path)

            data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)

            data_lowlight = (np.asarray(data_lowlight) / 255.0)
            data_lowlight = torch.from_numpy(data_lowlight).float()

            return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)
