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
    return img


def image_to_tensor(image):
    image = np.array(image)
    image = np.moveaxis(image, -1, 0)
    return torch.Tensor(image / 255.0)
