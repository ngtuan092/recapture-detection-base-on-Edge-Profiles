import numpy as np
import torch
import torch.nn as nn


class Threshold(nn.Module):
    def __init__(self, low_threshold: int = .05, high_threshold: int = .15) -> None:
        super().__init__()
        self.strong_pixel = 255
        self.weak_pixel = 25
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, img: torch.Tensor, hysteresis: bool = True) -> torch.Tensor:
        img = img[0, 0, :, :]
        highThreshold = img.max() * self.high_threshold
        lowThreshold = highThreshold * self.low_threshold

        strong_i, strong_j = torch.where(img >= highThreshold)
        zeros_i, zeros_j = torch.where(img < lowThreshold)
        weak_i, weak_j = torch.where(
            (img <= highThreshold) & (img >= lowThreshold))

        img[strong_i, strong_j] = self.strong_pixel
        img[weak_i, weak_j] = self.weak_pixel
        img[zeros_i, zeros_j] = 0

        if hysteresis:
            for i in range(1, img.shape[0] - 1):
                for j in range(1, img.shape[1] - 1):
                    if (img[i, j] == self.weak_pixel):
                        try:
                            if ((img[i + 1, j - 1] == self.strong_pixel) or (img[i + 1, j] == self.strong_pixel) or (img[i + 1, j + 1] == self.strong_pixel)
                                    or (img[i, j - 1] == self.strong_pixel) or (img[i, j + 1] == self.strong_pixel)
                                    or (img[i - 1, j - 1] == self.strong_pixel) or (img[i - 1, j] == self.strong_pixel) or (img[i - 1, j + 1] == self.strong_pixel)):
                                img[i, j] = self.strong_pixel
                            else:
                                img[i, j] = 0
                        except IndexError as e:
                            pass
        img = img.unsqueeze(0).unsqueeze(0)
        return img
