import numpy as np
from scipy import signal
import torch
import torch.nn.functional as F


def gaussian_kernel(kernel_size=21, std=3, channels=3):
    kernel1d = signal.gaussian(kernel_size, std=std).reshape(kernel_size, 1)
    kernel2d = np.outer(kernel1d, kernel1d)
    kernel4d = kernel2d[np.newaxis, np.newaxis, :, :]
    kernel4d = np.repeat(kernel4d, channels, axis=0)
    kernel4d = torch.from_numpy(kernel4d).float()
    kernel4d.requires_grad = False
    return kernel4d


def apply_kernel(images, kernel):
    """ Applies kernel to a batch of images. Kernel is applied per channel.
    Input:
      images: a batch of images in (N, C, H, W) format.
      kernel: a kernel to be applied to every image.
        Must have (out_channels, 1, H, W) shape.
    """
    kernel_dims = kernel.shape[2:]
    num_channels = images.shape[1]
    padding = tuple(map(int, (np.array(kernel_dims) - 1) / 2))
    out = F.conv2d(images, kernel, padding=padding, groups=num_channels)
    return out
