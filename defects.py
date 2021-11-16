import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from utils import get_random_color, get_random_ellipse


def add_spot(img, spot_bbox, color):
    draw = ImageDraw.Draw(img, 'RGBA')
    draw.ellipse(spot_bbox, fill=color)
    del draw
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
