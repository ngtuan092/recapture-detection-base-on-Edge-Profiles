import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonMaxSuppression(nn.Module):
    def _quantize_angle(self, theta: torch.Tensor):
        theta = theta * 180 / np.pi
        theta[theta < 0] += 180
        theta = torch.round(theta / 45) * 45
        return theta

    def _get_pixel(self, x: int, y: int):
        return self.img[0, 0, x, y]

    def _is_local_max(self, x: int, y: int, theta: int):
        if theta == 0 or theta == 180:
            return self._get_pixel(x, y) >= self._get_pixel(x, y + 1) and self._get_pixel(x, y) >= self._get_pixel(x, y - 1)
        elif theta == 45:
            return self._get_pixel(x, y) >= self._get_pixel(x + 1, y + 1) and self._get_pixel(x, y) >= self._get_pixel(x - 1, y - 1)
        elif theta == 90:
            return self._get_pixel(x, y) >= self._get_pixel(x + 1, y) and self._get_pixel(x, y) >= self._get_pixel(x - 1, y)
        elif theta == 135:
            return self._get_pixel(x, y) >= self._get_pixel(x + 1, y - 1) and self._get_pixel(x, y) >= self._get_pixel(x - 1, y + 1)
        else:
            raise Exception('Invalid angle')

    def forward(self, img: torch.Tensor, theta: torch.Tensor):
        # create new image
        self.img = img
        self.theta = self._quantize_angle(theta)

        new_img = torch.zeros_like(self.img)
        # iterate over image
        for x in range(1, self.img.shape[2] - 1):
            for y in range(1, self.img.shape[3] - 1):
                # get angle
                theta = self.theta[0, 0, x, y]
                # check if local max
                if self._is_local_max(x, y, theta):
                    new_img[0, 0, x, y] = self.img[0, 0, x, y]
        return new_img
