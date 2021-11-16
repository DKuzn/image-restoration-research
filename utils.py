import random as rd


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
