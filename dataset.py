import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random as rd
import os
from defects import add_random_spot


class ImageNet2012(Dataset):
    def __init__(self, root: str, paths_list: str):
        self.root = root
        self.paths = self._list_paths(paths_list)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = f'{self.root}{os.sep}' + self.paths[item].replace('/', os.sep).split(' ')[0] + '.JPEG'
        image = Image.open(path).convert('RGB')
        image = image.resize((256, 256))
        y_image = image
        for _ in range(rd.randint(0, 10)):
            y_image = add_random_spot(y_image)
        image = np.array(image)
        image = np.rollaxis(image, 2)
        y_image = np.array(y_image)
        y_image = np.rollaxis(y_image, 2)
        x = torch.Tensor(image / 255.0)
        y = torch.Tensor(y_image / 255.0)
        return y, x

    def _list_paths(self, paths_list: str):
        paths = open(paths_list, 'r').read().split('\n')
        if paths[-1] == '':
            paths = paths[:-1]
        rd.shuffle(paths)
        return paths


if __name__ == '__main__':
    ds = ImageNet2012('ImageNet2012\\test', 'test.txt')
    print(ds.paths[0])
    print(ds[0])
    print(torch.nn.MSELoss()(ds[0][0], ds[0][1]))
