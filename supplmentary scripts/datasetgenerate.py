import collections
import pathlib
import random
import os
import pickle
from typing import Dict, Tuple, Sequence

import cv2
from skimage.color import rgb2lab, lab2rgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable

def extract_dominant_colors(img):
    rgb_img = img.reshape(-1, 3)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 5
    ret, label, center = cv2.kmeans(np.float32(rgb_img), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    unique, counts = np.unique(label, return_counts=True)
    dominant_colors = center[unique].tolist()
    percentages = counts / sum(counts) * 100

    palette = []
    for color in dominant_colors:
        hex_code = "#{:02x}{:02x}{:02x}".format(*map(int, color))
        palette.append(hex_code)

    return palette


def viz_color_palette(hexcodes):

    palette = []
    for hexcode in hexcodes:
        rgb = np.array(list(int(hexcode.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
        palette.append(rgb)

    palette = np.array(palette)[np.newaxis, :, :]
    return palette

def augment_image(img, title, hue_shift):

    img_HSV = matplotlib.colors.rgb_to_hsv(img)
    a_2d_index = np.array([[1,0,0] for _ in range(img_HSV.shape[1])]).astype('bool')
    img_HSV[:, a_2d_index] = (img_HSV[:, a_2d_index] + hue_shift) % 1

    new_img = matplotlib.colors.hsv_to_rgb(img_HSV).astype(int)
    plt.imshow(new_img)
    plt.title(f"New {title} (in RGB)")
    plt.show()

    img = img.astype(np.float) / 255.0
    new_img = new_img.astype(np.float) / 255.0
    ori_img_LAB = rgb2lab(img)
    new_img_LAB = rgb2lab(new_img)
    new_img_LAB[:, :, 0] = ori_img_LAB[:, :, 0]
    new_img_augmented = (lab2rgb(new_img_LAB)*255.0).astype(int)
    plt.imshow(new_img_augmented)
    plt.title(f"New {title} (in RGB) with Fixed Luminance")
    plt.show()
    plt.close()

    return new_img_augmented

images = [f for f in os.listdir() if f.endswith(".jpg")]
for path in images:

    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    palette = viz_color_palette(extract_dominant_colors(img))

    hue_shift = random.random()
    augmented_image = augment_image(img, "Image", hue_shift)
    augmented_palette = augment_image(palette, "Palette", hue_shift)

    path_stem = path[:-4]

    cv2.imwrite(f'data/train/input/{path}', img)
    pickle.dump(palette, open(f'data/train/old_palette/{path_stem}.pkl', 'wb'))
    cv2.imwrite(f'data/train/output/{path}', augmented_image)
    pickle.dump(augmented_palette, open(f'data/train/new_palette/{path_stem}.pkl', 'wb'))
