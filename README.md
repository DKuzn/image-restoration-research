# Image Restoration Research

This repository contains the library to image resrotation with neural network. The libraty was developed during completion of my course work.

## How to train the network?

Install my [library](https://github.com/DKuzn/pytorch-training) to train PyTorch models and follow this example code:

```python
from pytorch_training import training
import torch
from torch.utils.data import DataLoader
from image_restoration.dataset import ImageNet2012
from image_restoration.model import RED10
from image_restoration.metrics import psnr
from image_restoration.defects import spots


if __name__ == '__main__':
    ds_train = ImageNet2012('ImageNet2012\\test', 'test.txt', transform=spots, size=256, transform_kwargs={'max_count': 10})
    dl_train = DataLoader(ds_train, batch_size=20)
    ds_test = ImageNet2012('ImageNet2012\\val', 'val.txt', transform=spots, size=256, transform_kwargs={'max_count': 10})
    dl_test = DataLoader(ds_test, batch_size=10)
    model = RED10()

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)

    training(dl_train, dl_test, model, optimizer, loss_function, psnr, epochs=4)
```

## How to run testing notebook?

Install additional dependencies:
 - matplotlib
 - scikit-image

Download ImageNet [dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge/data?select=imagenet_object_localization_patched2019.tar.gz) and put weights from releases page to "weights" directory.