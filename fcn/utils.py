import copy
import os
import random

import numpy as np
from albumentations import ElasticTransform, GaussianBlur, MedianBlur
from scipy.ndimage import zoom


def random_crop(image, label, output_size):
    (w, h, d) = image.shape
    w1 = np.random.randint(0, w - output_size[0])
    h1 = np.random.randint(0, h - output_size[1])

    label = label[w1:w1 + output_size[0], h1:h1 + output_size[1]]
    image = image[w1:w1 + output_size[0], h1:h1 + output_size[1], :]
    return image, label


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_noise(image, label, mu=0, sigma=0.1):
    noise = np.clip(
        sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*sigma, 2*sigma)
    noise = noise + mu
    image = image + noise
    return image, label


def random_elastic_transform(image, label):
    size = random.randrange(100, 500)
    aug = ElasticTransform(p=1, alpha=size, sigma=size *
                           0.05, alpha_affine=size * 0.03)
    augmented = aug(image=image, mask=label)
    image = augmented['image']
    label = augmented['mask']
    return image, label


def random_blur(image, label):
    limit = random.randrange(10, 50)
    if random.random() > 0.5:
        aug = GaussianBlur(blur_limit=limit, p=1)
    else:
        aug = MedianBlur(blur_limit=limit, p=1)
    augmented = aug(image=image)
    image = augmented['image']
    return image, label


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
