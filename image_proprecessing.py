import os

import cv2
import numpy as np
from PIL import Image


def image_crop(data_path, save_path, patch_size=(400, 400), strich_hw=(60, 60)):
    print(data_path, save_path)
    image_type = ["image", "label"]
    image_list = os.listdir(data_path + "/" + image_type[0])
    for i in image_type:
        if not os.path.exists(save_path + "/" + i):
            os.makedirs(save_path + "/" + i)
    for im in image_list:
        image = np.array(Image.open(
            data_path + "/" + image_type[0] + "/" + im))
        if data_path.startswith("data/Test"):
            label = np.array(Image.open(
                data_path + "/" + image_type[1] + "/" + im.split(".")[0] + ".png"))
        else:
            label = np.array(Image.open(
                data_path + "/" + image_type[1] + "/" + im))
        stridew, strideh = patch_size
        winW, winH = strich_hw
        num = 0
        shapeW, shapeH, channel = image.shape
        for h in range((shapeH-winH) // strideh + 1):
            for w in range((shapeW-winW) // stridew + 1):
                img = image[w*stridew:w*stridew+winW, h*strideh:h*strideh+winH]
                lab = label[w*stridew:w*stridew+winW, h*strideh:h*strideh+winH]
                num = num + 1
                cv2.imwrite("{}/image/{}_{}.png".format(
                    save_path, im.split(".")[0], num), img)
                cv2.imwrite("{}/label/{}_{}.png".format(
                    save_path, im.split(".")[0], num), lab)


if __name__ == "__main__":
    # image_crop("data/Train", "data/preprocessing/Train")
    image_crop("data/Test", "data/preprocessing/Test", patch_size=(250, 250), strich_hw=(250, 250))
