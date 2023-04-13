#!/usr/bin/env python
# coding: utf-8
import collections
import pathlib
import random
import os
import pickle
from typing import Dict, Tuple, Sequence

import cv2
import xlwings.constants
from skimage.color import rgb2lab, lab2rgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import PIL.Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from functools import partial
from torch.utils.data import Dataset, DataLoader
import pathlib
from tkinter import *
from tkinter import colorchooser

device = "cuda" if torch.cuda.is_available() else "cpu"


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.InstanceNorm2d(out_channels))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """

    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()

        # convolutional
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.norm1_1 = nn.InstanceNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # residual blocks
        self.res1 = ResNetLayer(64, 128, block=ResNetBasicBlock, n=1)
        self.res2 = ResNetLayer(128, 256, block=ResNetBasicBlock, n=1)
        self.res3 = ResNetLayer(256, 512, block=ResNetBasicBlock, n=1)

    def forward(self, x):
        x = F.relu(self.norm1_1(self.conv1_1(x)))
        c4 = self.pool1(x)
        c3 = self.res1(c4)
        c2 = self.res2(c3)
        c1 = self.res3(c2)
        return c1, c2, c3, c4


#
# # In[8]:
#
#
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
    )


class RecoloringDecoder(nn.Module):
    # c => (bz, channel, h, w)
    # [Pt, c1]: (18 + 512) -> (256)
    # [c2, d1]: (256 + 256) -> (128)
    # [Pt, c3, d2]: (18 + 128 + 128) -> (64)
    # [Pt, c4, d3]: (18 + 64 + 64) -> 64
    # [Illu, d4]: (1 + 64) -> 3

    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up_4 = double_conv(12 + 512, 256)
        self.dconv_up_3 = double_conv(256 + 256, 128)
        self.dconv_up_2 = double_conv(12 + 128 + 128, 64)
        self.dconv_up_1 = double_conv(12 + 64 + 64, 64)
        self.conv_last = nn.Conv2d(1 + 64, 3, 3, padding=1)

    def forward(self, c1, c2, c3, c4, target_palettes_1d, illu):
        bz, h, w = c1.shape[0], c1.shape[2], c1.shape[3]
        target_palettes = torch.ones(bz, 12, h, w).float().to(device)
        target_palettes = target_palettes.reshape(h, w, bz * 12) * target_palettes_1d
        target_palettes = target_palettes.permute(2, 0, 1).reshape(bz, 12, h, w)

        # concatenate target_palettes with c1
        x = torch.cat((c1.float(), target_palettes.float()), 1)
        x = self.dconv_up_4(x)
        x = self.upsample(x)

        # concatenate c2 with x
        x = torch.cat([c2, x], dim=1)
        x = self.dconv_up_3(x)
        x = self.upsample(x)

        # concatenate target_palettes and c3 with x
        bz, h, w = x.shape[0], x.shape[2], x.shape[3]
        target_palettes = torch.ones(bz, 12, h, w).float().to(device)
        target_palettes = target_palettes.reshape(h, w, bz * 12) * target_palettes_1d
        target_palettes = target_palettes.permute(2, 0, 1).reshape(bz, 12, h, w)
        x = torch.cat([target_palettes.float(), c3, x], dim=1)
        x = self.dconv_up_2(x)
        x = self.upsample(x)

        # concatenate target_palettes and c4 with x
        bz, h, w = x.shape[0], x.shape[2], x.shape[3]
        target_palettes = torch.ones(bz, 12, h, w).float().to(device)
        target_palettes = target_palettes.reshape(h, w, bz * 12) * target_palettes_1d
        target_palettes = target_palettes.permute(2, 0, 1).reshape(bz, 12, h, w)
        x = torch.cat([target_palettes.float(), c4, x], dim=1)
        x = self.dconv_up_1(x)
        x = self.upsample(x)
        illu = illu.view(illu.size(0), 1, illu.size(1), illu.size(2))
        x = torch.cat((x, illu), dim=1)
        x = self.conv_last(x)
        return x


def get_illuminance(img):
    """
    Get the luminance of an image. Shape: (h, w)
    """
    img = img.permute(1, 2, 0)  # (h, w, channel)
    img = img.numpy()
    img = img.astype(np.float) / 255.0
    img_LAB = rgb2lab(img)
    img_L = img_LAB[:, :, 0]  # luminance  # (h, w)
    return torch.from_numpy(img_L)


class ColorTransferDataset(Dataset):
    def __init__(self, data_folder, transform):
        super().__init__()
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        output_folder = self.data_folder / "output"
        return len(list(output_folder.glob("*")))

    def __getitem__(self, idx):
        input_img_folder = self.data_folder / "input"
        old_palette = self.data_folder / "old_palette"
        new_palette = self.data_folder / "new_palette"
        output_img_folder = self.data_folder / "output"
        files = list(output_img_folder.glob("*"))

        f = files[idx]
        ori_image = transform(cv2.imread(str(input_img_folder / f.name)))
        new_image = transform(cv2.imread(str(output_img_folder / f.name)))
        illu = get_illuminance(ori_image)

        new_palette = pickle.load(open(str(new_palette / f.stem) + '.pkl', 'rb'))
        new_palette = new_palette[:, :6, :].ravel() / 255.0

        old_palette = pickle.load(open(str(old_palette / f.stem) + '.pkl', 'rb'))
        old_palette = old_palette[:, :6, :].ravel() / 255.0

        ori_image = ori_image.double()
        new_image = new_image.double()
        illu = illu.double()
        new_palette = torch.from_numpy(new_palette).double()
        old_palette = torch.from_numpy(old_palette).double()

        return ori_image, new_image, illu, new_palette, old_palette


def viz_color_palette(hexcodes):
    """
    visualize color palette
    """
    hexcodes = list(hexcodes)
    while len(hexcodes) < 6:
        hexcodes = hexcodes + hexcodes
    hexcodes = hexcodes[:6]

    palette = []
    for hexcode in hexcodes:
        rgb = np.array(list(int(hexcode.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)))
        palette.append(rgb)

    palette = np.array(palette)[np.newaxis, :, :]
    return palette


def viz_image_ori_new_out(ori, palette, new, out):
    """
    visualize original image, input palette, true new image, and output image from the model.
    """
    ori = ori.detach().cpu().numpy()
    new = new.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    palette = palette.detach().cpu().numpy()

    plt.imshow(np.transpose(ori, (1, 2, 0)), interpolation='nearest')
    plt.title("Original Image")
    plt.show()

    palette = palette.reshape((1, 4, 3))
    plt.imshow(palette, interpolation='nearest')
    plt.title("Palette")
    plt.show()

    # plt.imshow((np.transpose(out, (1,2,0)) * 255).astype(np.uint8))
    plt.imshow((np.transpose(out, (1, 2, 0))))
    plt.title("Output Image")
    plt.show()

    plt.imshow(np.transpose(new, (1, 2, 0)), interpolation='nearest')
    plt.title("True Image")
    plt.show()


def viz_image_ori_new_out2(ori, palette, out):
    """
    visualize original image, input palette, true new image, and output image from the model.
    """
    ori = ori.detach().cpu().numpy()
    # new = new.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    palette = palette.detach().cpu().numpy()

    plt.imshow(np.transpose(ori, (1, 2, 0)), interpolation='nearest')
    plt.title("Original Image")
    plt.show()

    palette = palette.reshape((1, 4, 3))
    plt.imshow(palette, interpolation='nearest')
    plt.title("Palette")
    plt.show()

    # plt.imshow((np.transpose(out, (1,2,0)) * 255).astype(np.uint8))
    plt.imshow((np.transpose(out, (1, 2, 0))))
    plt.title("Output Image")
    plt.show()

    fig = plt.imshow((np.transpose(out, (1, 2, 0))))
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('pict.png', bbox_inches='tight', pad_inches=0)

    image = PIL.Image.open('pict.png')
    new_image = image.resize((256, 256))
    new_image.save('pict256.png')

    os.remove('pict.png')


    # plt.imshow(np.transpose(new, (1, 2, 0)), interpolation='nearest')
    # plt.title("True Image")
    # plt.show()


def choose_color():
    # variable to store hexadecimal code of color
    color_code = colorchooser.askcolor(title="Choose color")
    return list(color_code[0])


# root = Tk()
# button = Button(root, text="Select color",
#                 command=choose_color)
# button.pack()
# root.geometry("300x300")
# root.mainloop()


bz = 16
epoches = 100
lr = 0.0002

# pre-processsing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# dataset and dataloader
train_data = ColorTransferDataset(pathlib.Path("data/train"), transform)
train_loader = DataLoader(train_data, batch_size=bz)
# train_data.__getitem__(0)

# create model, criterion and optimzer
FE = FeatureEncoder().float().to(device)
RD = RecoloringDecoder().float().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(list(FE.parameters()) + list(RD.parameters()), lr=lr, weight_decay=4e-3)

state = torch.load("saved_models/FE_RD.pth")
FE = FeatureEncoder().float().to(device)
RD = RecoloringDecoder().float().to(device)
FE.load_state_dict(state['FE'])
RD.load_state_dict(state['RD'])
optimizer.load_state_dict(state['optimizer'])

k = 4
user_palette = []
for i in range(k):
    user_palette.append(choose_color())

print(user_palette)
user_palette = np.array(user_palette)[np.newaxis, :, :]
print(user_palette)
user_palette = user_palette[:, :6, :].ravel() / 255.0
print(user_palette)
user_palette = torch.from_numpy(user_palette).double()
print(user_palette)

og_image = cv2.imread("image.jpg")
og_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2RGB)
og_image = transform(og_image)
og_image = og_image.double()
illum = get_illuminance(og_image)
og_image = og_image.unsqueeze(0)
illum = illum.double()
illum = illum.unsqueeze(0)
flat_palette = user_palette.flatten()

c1, c2, c3, c4 = FE.forward(og_image.float().to(device))
out = RD.forward(c1, c2, c3, c4, flat_palette.float().to(device), illum.float().to(device))
idx = 3
viz_image_ori_new_out2(og_image[0], user_palette, out[0])
