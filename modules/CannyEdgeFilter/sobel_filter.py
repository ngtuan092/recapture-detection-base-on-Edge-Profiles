import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelFilter(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.kernel = self._create_kernel(size)

    def _create_kernel(self, k: int = 5):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[sobel_2D_denominator == 0] = 1
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D

    def forward(self, x: torch.Tensor):
        # add channel dimension
        assert x.dim() == 4, "Input must be a 4D tensor"
        # create kernel
        kernel = torch.from_numpy(
            self.kernel).float().unsqueeze(0).unsqueeze(0)
        # convolve
        x = F.conv2d(x, kernel, padding=self.size // 2)
        y = F.conv2d(x, kernel.transpose(2, 3), padding=self.size // 2)
        G = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)
        G = G / torch.max(G) * 255

        return G, theta



