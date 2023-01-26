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
        self.weak_pixel = 75
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
    from PIL import Image
    import os
    img = Image.open(os.path.join('images', 'IMG_1129.JPG'))

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.GaussianBlur(5, 1)
    ])

    img = transform(img)
    img.unsqueeze_(0)
    img = filter(img)
    img = img.squeeze(0)
    img = img.squeeze(0)

    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()
