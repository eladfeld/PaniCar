from PIL import Image
import numpy as np

import requests
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2 as cv
from torchvision.utils import draw_bounding_boxes
import torch
import torchvision
import random


# flare intensity should be between 0.8-2
# opacity level should be between 0.8-2
# channel is the color
def make_light(x0, y0, x1, y1, img_shape, channel=2, flare_radius=0.5, opacity_level=1):
    r_w = 100
    r_h = 20

    # flare parameters
    e_c_min = 4
    e_c_max = 260
    s_c_min = 30
    s_c_max = 170
    e_w_min = 30
    e_w_max = 3600
    s_w_min = 30
    s_w_max = 170

    # generate placement
    x = int(x0 + (x1 - x0) // 2)  # middle of box (respect to image)
    y = int(y1 - (y1 - y0) * 0.9)  # 80% from bottom of box (respect to image)

    # make color layer and add light
    layer_c = np.zeros(img_shape)
    layer_c[(y - r_h):(y + r_h), (x - r_w):(x + r_w), channel] = 255 * (
            flare_radius ** 2 * (e_c_max - e_c_min) + e_c_min)  # RANGE 255*4:260
    # blur color light
    layer_c[:, :, channel] = gaussian_filter(layer_c[:, :, channel],
                                             sigma=(flare_radius ** 0.7 * (s_c_max - s_c_min) + s_c_min),
                                             mode='constant')

    # make white layer and add light
    layer_w = np.zeros(img_shape)
    layer_w[(y - r_h):(y + r_h), (x - r_w):(x + r_w), :] = 255 * (flare_radius ** 2 * (e_w_max - e_w_min) + e_w_min)
    # blur white light
    layer_w = gaussian_filter(layer_w, sigma=(flare_radius * (s_w_max - s_w_min) + s_w_min), mode='constant')
    print(np.max(layer_w))

    return np.clip(opacity_level * (layer_c + layer_w), 0, 255)


def make_light2(x, y, img_shape, option):
    r_w, r_h = option_dict[option]

    channel = 0 if np.random.rand() > 0.5 else 2

    # make color layer and add light
    layer_c = np.zeros(img_shape)
    layer_c[(y - r_h):(y + r_h), (x - r_w):(x + r_w), channel] = 255 * 65
    # blur color light
    layer_c[:, :, channel] = gaussian_filter(layer_c[:, :, channel], sigma=100, mode='constant')

    # make white layer and add light
    layer_w = np.zeros(img_shape)
    layer_w[(y - r_h):(y + r_h), (x - r_w):(x + r_w), :] = 255 * 900
    # blur white light
    layer_w = gaussian_filter(layer_w, sigma=100, mode='constant')
    # print(np.max(layer_w))

    return np.clip(layer_c + layer_w, 0, 255)


# takes img in np.array format, returns in numpy array format
# bbox coords is in pixels
def aug_image(img, bbox):
    xmin, ymin, xmax, ymax = bbox
    opacity_min = 1
    # opacity_max = 2
    radius_min = 0.3
    # radius_max = 2

    opacity = np.abs(np.random.randn()) + opacity_min
    radius = np.abs(np.random.randn()) * .8 + radius_min
    color = 0 if np.random.rand() > 0.5 else 2  # choose red or blue at random

    try:
        layer = make_light(int(xmin), int(ymin), int(xmax), int(ymax), img.shape, channel=color, flare_radius=radius,
                           opacity_level=opacity)
        return np.clip(img + layer, 0, 255).astype(int)
    except:
        print("Error in flare augmentation")
        return img


# takes img in np.array format, returns in numpy array format
# bbox coords is in pixels
def aug_image2(img, bbox, option):
    x_co = random.randint(0, img.shape[1])
    y_co = random.randint(0, img.shape[0])
    layer = make_light2(x_co, y_co, img.shape, option)
    return np.clip(img + layer, 0, 255).astype(int)


# takes img in np.array format, returns in numpy array format
# bbox coords is in pixels
def aug_image(Img, bbox):
    xmin, ymin, xmax, ymax = bbox
    opacity_min = 1
    # opacity_max = 2
    radius_min = 0.5
    # radius_max = 2

    opacity = np.abs(np.random.randn()) * .1 + opacity_min
    radius = np.abs(np.random.randn()) * .1 + radius_min
    color = 0 if np.random.rand() > 0.5 else 2  # choose red or blue at random

    try:
        layer = make_light(int(xmin), int(ymin), int(xmax), int(ymax), Img.shape, channel=color, flare_radius=radius,
                           opacity_level=opacity)
        return np.clip(Img + layer, 0, 255).astype(int)
    except:
        print("Error in flare augmentation")
        return Img


def draw_bbox(img, bbox):
    img = torch.from_numpy(img)
    box = torch.tensor(bbox, dtype=torch.int)
    box = box.unsqueeze(0)
    img = draw_bounding_boxes(img, box, width=5,
                              colors="green",
                              fill=True)
    img = torchvision.transforms.ToPILImage()(img)
    img.show()



option_dict = {1: (100, 20), 2:(10,15)}
def main(image_name, bbox, output_image, option):
    img = cv.imread(image_name)
    Img_aug = aug_image2(img, bbox, option)
    # draw_bbox(Img_aug, bbox)
    cv.imwrite(output_image, Img_aug)
