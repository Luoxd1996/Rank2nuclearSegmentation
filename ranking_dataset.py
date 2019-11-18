import copy
import os
import random

import cv2
import numpy as np
import torch
from albumentations import ElasticTransform, GaussianBlur, MedianBlur
from PIL import Image
from scipy.ndimage import zoom
from torch.utils import data
from torchvision import datasets, models, transforms


def data_geneartor(image):
    if random.random() > 0.5:
        image = random_rot_flip(image)
    if random.random() > 0.5:
        image = random_elastic_transform(image)

    # if random.random() > 0.5:
    #     image = random_color_jittering(image)

    if random.random() > 0.5:
        image = random_blur(image)

    if random.random() > 0.5:
        image = random_noise(image)
    image = image / image.max()
    return image


def ranking_data_geneartor(x, ranking_crops=5, cropped_stride_x=25, cropped_stride_y=25, image_shape=[256, 256]):
    x_raw = x.copy()
    shapex, shapey, _ = x_raw.shape
    mini_batch = []
    for i in range(ranking_crops):
        x_b, x_e, y_b, y_e = i * \
            cropped_stride_x, (shapex-i*cropped_stride_x), i * \
            cropped_stride_y, (shapey-i*cropped_stride_y)
        img = x_raw[x_b:x_e, y_b:y_e, :]
        img = zoom_image(img, image_shape)
        img = data_geneartor(img)
        img = img / img.max()
        img = img.transpose((2, 0, 1))
        mini_batch.append(img)
    return np.array(mini_batch)


class RankData(data.Dataset):

    def __init__(self, data_list="Train", aug=False, patch_size=[500, 500]):
        self.data_list = data_list
        self.aug = aug
        self.patch_size = patch_size
        self._read_path()

    def __len__(self):
        return len(self.data_path)

    def _read_path(self):
        self.data_path = []
        img_file = os.listdir("data/{}/image".format(self.data_list))
        self.data_path = [
            "data/{}/image/".format(self.data_list) + i for i in img_file]

    def __getitem__(self, index):
        x = np.array(Image.open(self.data_path[index]).convert("RGB"))
        x = random_crop(x, output_size=self.patch_size)
        mini_batch = ranking_data_geneartor(x, image_shape=[self.patch_size[0]-100, self.patch_size[1]-100])
        label = [4, 3, 2, 1, 0]
        random.shuffle(label)
        mini_batch = mini_batch[label, :, :, :]
        mini_batch = torch.from_numpy(mini_batch).float()
        label = torch.from_numpy(np.array(label))
        return mini_batch, label


def random_crop(image, output_size):
    (w, h, d) = image.shape
    w1 = np.random.randint(0, w - output_size[0])
    h1 = np.random.randint(0, h - output_size[1])
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], :]
    return image


def random_rot_flip(image):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    return image


def random_noise(image, mu=0, sigma=0.1):
    noise = np.clip(
        sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*sigma, 2*sigma)
    noise = noise + mu
    image = image + noise
    return image


def random_elastic_transform(image):
    size = random.randrange(100, 500)
    aug = ElasticTransform(p=1, alpha=size, sigma=size *
                           0.05, alpha_affine=size * 0.03)
    augmented = aug(image=image)
    image = augmented['image']
    return image


def random_blur(image):
    limit = random.randrange(10, 50)
    if random.random() > 0.5:
        aug = GaussianBlur(blur_limit=limit, p=1)
    else:
        aug = MedianBlur(blur_limit=limit, p=1)
    augmented = aug(image=image)
    image = augmented['image']
    return image


def random_color_jittering(image):
    rgb = [0, 1, 2]
    random.shuffle(rgb)
    image = image[:, :, rgb]

    if random.random() > 0.5:
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]

        rand1 = random.randrange(-20, 20)
        rand2 = random.randrange(-20, 20)
        rand3 = random.randrange(-20, 20)

        image = np.dstack((
            np.roll(R, rand1, axis=0),
            np.roll(G, rand2, axis=1),
            np.roll(B, rand3, axis=0)
        ))
    return image


def zoom_image(data, size=[400, 400]):
    """
    reshape image to inputs pixels
    """
    if len(data.shape) == 3:
        x, y, z = data.shape
        zoomed_image = zoom(data, (size[0] / x, size[1] / y, 1))
    if len(data.shape) == 2:
        x, y = data.shape
        zoomed_image = zoom(data, (size[0] / x, size[1] / y))
    return zoomed_image
