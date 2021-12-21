import torch


def psnr(source_tesnor, tensor):
    error = torch.nn.MSELoss()(source_tesnor, tensor)
    return 10 * torch.log10(1 / error)


if __name__ == '__main__':
    t1 = torch.tensor([0.5, 0.6, 0.1, 0.2])
    t2 = t1 * 1.0000001
    print(t1)
    print(t2)
    print(psnr(t1, t2))