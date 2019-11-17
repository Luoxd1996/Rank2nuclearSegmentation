import copy
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import datasets, models, transforms

from utils import (random_blur, random_crop, random_elastic_transform,
                   random_noise, random_rot_flip, zoom_image)


def data_geneartor(image, label):
    if random.random() > 0.5:
        image, label = random_rot_flip(image, label)
    if random.random() > 0.5:
        image, label = random_elastic_transform(image, label)

    if random.random() > 0.5:
        image, label = random_blur(image, label)

    if random.random() > 0.5:
        image, label = random_noise(image, label)
    image = image / image.max()
    label[label > 0] = 1
    return image, label


def multi_tasks_data_geneartor(x, y, ranking_crops=5, cropped_stride_x=25, cropped_stride_y=25, image_shape=[256, 256]):
    x_raw = x.copy()
    y_raw = y.copy()
    shapex, shapey, _ = x_raw.shape
    mini_batch_img = []
    mini_batch_lab = []
    for i in range(ranking_crops):
        x_b, x_e, y_b, y_e = i * \
            cropped_stride_x, (shapex-i*cropped_stride_x), i * \
            cropped_stride_y, (shapey-i*cropped_stride_y)
        img = x_raw[x_b:x_e, y_b:y_e, :]
        lab = y_raw[x_b:x_e, y_b:y_e, :]
        img = zoom_image(img, image_shape)
        lab = zoom_image(lab, image_shape)

        img, lab = data_geneartor(img, lab)
        img = img / img.max()
        img = img.transpose((2, 0, 1))
        mini_batch_img.append(img)
        mini_batch_lab.append(lab)
    return np.array(mini_batch_img), np.array(mini_batch_lab)


class MultiTaskData(data.Dataset):

    def __init__(self, data_list="Train", aug=False):
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

        img, lab = random_crop(img, lab, output_size=[500, 500])
        mini_batch_img, mini_batch_lab = multi_tasks_data_geneartor(
            img, lab, image_shape=[400, 400])

        classes = [4, 3, 2, 1, 0]
        random.shuffle(classes)
        batch_img = mini_batch_img[classes, :, :, :]
        batch_lab = mini_batch_lab[classes, :, :, :]

        img, lab = torch.from_numpy(batch_img).float(
        ), torch.from_numpy(np.array([batch_lab])).float()
        return img, lab, classes
