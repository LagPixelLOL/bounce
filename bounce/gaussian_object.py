import numpy as np
from .base_object import BaseObject

class GaussianObject(BaseObject):

    def __init__(self, pos2d, vel2d, std):
        super().__init__(pos2d, vel2d)
        self.std = std

    # Gaussian with (0, 1].
    def __call__(self, x, y):
        return np.exp(-((x - self.pos2d[0]) ** 2 + (y - self.pos2d[1]) ** 2) / (2 * self.std ** 2))
