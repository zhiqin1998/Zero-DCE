import glob
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

random.seed(1143)

def populate_train_list(lowlight_images_paths):
    train_list = []

    for lowlight_images_path in lowlight_images_paths.split(','):
        lowlight_images_path = lowlight_images_path.strip()
        image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")

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
