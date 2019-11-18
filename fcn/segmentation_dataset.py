import copy
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import datasets, models, transforms
from utils import random_blur, random_crop, random_elastic_transform, random_noise, random_rot_flip
from skimage import filters
from skimage.morphology import remove_small_objects, dilation, erosion, square


def data_geneartor(image, label, patch_size=[400, 400]):
    image, label = random_crop(image, label, patch_size)
    if random.random() > 0.5:
        image, label = random_rot_flip(image, label)
    if random.random() > 0.5:
        image, label = random_elastic_transform(image, label)

    if random.random() > 0.5:
        image, label = random_blur(image, label)

    if random.random() > 0.5:
        image, label = random_noise(image, label)

    if random.random() > 0.5:
        channel = [0, 1, 2]
        random.shuffle(channel)
        image = image[:, :, channel]
    image = image / image.max()
    label[label > 0] = 1
    return image, label


class SegmentationData(data.Dataset):

    def __init__(self, data_list="Train", patch_size=[400, 400], aug=False):
        self.patch_size = patch_size
        self.data_list = data_list
        self.aug = aug
        self._read_path()

    def __len__(self):
        return len(self.data_path)

    def _read_path(self):
        self.data_path = []
        img_file = os.listdir(
            "data/{}/image".format(self.data_list))
        self.data_path = [
            "data/{}/image/".format(self.data_list) + i for i in img_file]

    def __getitem__(self, index):
        img = np.array(Image.open(self.data_path[index]))
        lab = np.array(Image.open(
            self.data_path[index].replace("image", "label").replace("Slide", "GT")).convert("L"))
        img, lab= data_geneartor(
            img, lab, patch_size=self.patch_size)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float(), 
        lab = torch.from_numpy(np.array([lab])).float()
        return img, lab




class ContourSegmentationData(data.Dataset):

    def __init__(self, data_list="Train", patch_size=[400, 400], aug=False):
        self.patch_size = patch_size
        self.data_list = data_list
        self.aug = aug
        self._read_path()

    def __len__(self):
        return len(self.data_path)

    def _read_path(self):
        self.data_path = []
        img_file = os.listdir(
            "data/{}/image".format(self.data_list))
        self.data_path = [
            "data/{}/image/".format(self.data_list) + i for i in img_file]

    def __getitem__(self, index):
        img = np.array(Image.open(self.data_path[index]))
        lab = np.array(Image.open(
            self.data_path[index].replace("image", "label").replace("Slide", "GT")).convert("L"))
        img, lab = data_geneartor(
            img, lab, patch_size=self.patch_size)
        marker = erosion(lab, square(3))
        contour = filters.scharr(lab)
        contour = dilation(contour, square(1))
        contour[contour>0]=1

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        lab = torch.from_numpy(np.array([lab])).float()
        marker = torch.from_numpy(np.array([marker])).float()
        contour = torch.from_numpy(np.array([contour])).float()
        return img, lab, marker, contour