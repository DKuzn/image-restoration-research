import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from image_restoration.utils import get_random_color, get_random_ellipse, get_random_scratch
import random as rd


def add_spot(img, spot_bbox, color):
    draw = ImageDraw.Draw(img, 'RGBA')
    draw.ellipse(spot_bbox, fill=color)
    del draw
    return img


def add_random_spot(img, seed=None):
    spot_bbox = get_random_ellipse(img.size, seed)
    color = get_random_color(seed)
    spoted_image = add_spot(img, spot_bbox, color)
    return spoted_image


def add_scratch(img, coords, color):
    draw = ImageDraw.Draw(img, 'RGBA')
    draw.line(coords, fill=color)
    del draw
    return img


def add_random_scratch(img, seed=None):
    coords = get_random_scratch(img.size, seed)
    color = get_random_color(seed)
    lined_image = add_scratch(img, coords, color)
    return lined_image


def gamma_color_transform(img, gamma):
    image = np.array(img)
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)
    return Image.fromarray(gamma_corrected)


def global_color_defect(img, seed=None):
    if seed is not None:
        rd.seed(seed)
    
    image = np.array(img)
    for i in range(3):
        image[:, :, i] = np.uint8(image[:, :, i] * rd.uniform(0, 1))
    return Image.fromarray(image)


def spots(img, max_count, seed=None):
    for _ in range(rd.randint(0, max_count)):
        img = add_random_spot(img, seed)
    return img


def scratches(img, max_count, seed=None):
    for _ in range(rd.randint(0, max_count)):
        img = add_random_scratch(img, seed)
    return img


def color(img, seed=None):
    if seed is not None:
        rd.seed(seed)

    if rd.randint(0, 4) != 1:
        img = global_color_defect(img, seed)

        if rd.randint(0, 1) == 0:
            img = gamma_color_transform(img, 0.5)

    return img


if __name__ == '__main__':
    img = Image.open('E:\\Projects\\image-repairing\\ImageNet2012\\test\\ILSVRC2012_test_00000002.JPEG').convert('RGB')
    bbox = get_random_ellipse(img.size)
    color = get_random_color()
    print(bbox)
    print(color)
    spoted_image = add_spot(img, bbox, color)
    plt.imshow(spoted_image)
    plt.show()
