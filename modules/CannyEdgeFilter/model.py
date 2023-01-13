import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from modules.CannyEdgeFilter.sobel_filter import SobelFilter
from modules.CannyEdgeFilter.non_max_suppresion import NonMaxSuppression
from modules.CannyEdgeFilter.threshold import Threshold


class CannyEdgeFilter(nn.Module):
    def __init__(self, size: int, low_threshold: int = .05, high_threshold: int = .15, sigma: int = 1):
        super().__init__()
        self.size = size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.strong_pixel = 255
        self.weak_pixel = 25
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
        x = transforms.GaussianBlur(self.size, self.sigma)(x)
        print(x.shape)
        sobel = SobelFilter(self.size)
        print(x.shape)
        x = sobel(x)
        nms = NonMaxSuppression()
        x = nms(x[0], x[1])
        print(x.shape)
        threshold = Threshold(self.low_threshold, self.high_threshold)
        x = threshold(x)
        return x


if __name__ == '__main__':

    filter = CannyEdgeFilter(5)

    x = torch.rand(1, 1, 100, 100)
    x = filter(x)
    print(x.shape)
