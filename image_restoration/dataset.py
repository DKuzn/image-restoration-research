import torch
from torch.utils.data import Dataset
from PIL import Image
import random as rd
import os
import copy
from image_restoration.utils import image_to_tensor
from typing import Callable


class ImageNet2012(Dataset):
    def __init__(self, root: str, paths_list: str, transform: Callable = None, size: int = None, transform_kwargs = {}):
        self.root = root
        self.paths = self._list_paths(paths_list)
        self.transform = transform
        self.size = size
        self.kwargs = transform_kwargs

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        ds_path = self.paths[item].replace('/', os.sep).split(' ')[0]
        path = f'{self.root}{os.sep}{ds_path}.JPEG'
        image = Image.open(path).convert('RGB')
        if self.size is not None:
            image = image.resize((self.size, self.size))
        y_image = copy.deepcopy(image)
        if self.transform is not None:
            image = self.transform(image, **self.kwargs)
        x = image_to_tensor(image)
        y = image_to_tensor(y_image)
        return x, y

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
