# image_restoration/utils.py
#
# Copyright (C) 2021 Дмитрий Кузнецов
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import random as rd
import numpy as np
import torch


def get_random_color(seed=None):
    if seed is not None:
        rd.seed(seed)

    color = []
    for i in range(4):
        color.append(rd.randint(0, 255))
    return tuple(color)


def get_random_ellipse(img_size, seed=None):
    if seed is not None:
        rd.seed(seed)

    ellipse = []
    ellipse.append(rd.randint(0, img_size[0]))
    ellipse.append(rd.randint(0, img_size[1]))
    ellipse.append(rd.randint(ellipse[0], img_size[0]))
    ellipse.append(rd.randint(ellipse[1], img_size[1]))

    return ellipse


def get_random_scratch(img_size, seed=None):
    if seed is not None:
        rd.seed(seed)

    line = []
    for _ in range(2):
        line.append(rd.randint(0, img_size[0]))
        line.append(rd.randint(0, img_size[1]))

    return line


def tensor_to_image(tensor):
    img = tensor.detach().numpy()
    img = np.moveaxis(img, 0, -1)
    return np.uint8(img * 255.0)


def image_to_tensor(image):
    image = np.array(image)
    image = np.moveaxis(image, -1, 0)
    return torch.Tensor(image / 255.0)
