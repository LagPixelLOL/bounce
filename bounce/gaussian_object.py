import numpy as np
from .base_object import BaseObject

class GaussianObject(BaseObject):

    def __init__(self, pos2d, vel2d=np.zeros(2), std=1.0, color3d=np.array([255, 255, 255], dtype=np.int16), frozen=False):
        super().__init__(pos2d, vel2d, frozen)
        self.std = std
        self.color3d = color3d

    # Gaussian with (0, 1].
    def __call__(self, x, y):
        return np.exp(-((x - self.pos2d[0]) ** 2 + (y - self.pos2d[1]) ** 2) / (2 * self.std ** 2))

    def render(self, x_matrix, y_matrix):
        return np.round(np.expand_dims(np.clip(self(x_matrix, y_matrix), 0, 1), -1).repeat(3, -1) * self.color3d).astype(np.int16)
