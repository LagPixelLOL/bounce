import numpy as np
from .utils import unit_diff, overlapped_volume

class BaseObject:

    def __init__(self, pos2d, vel2d):
        self.pos2d = pos2d
        self.vel2d = vel2d
        self.reset_acc()

    def __call__(self, x, y):
        raise NotImplementedError

    def step(self, t):
        self.vel2d += self.acc2d * t
        self.pos2d += self.vel2d * t
        self.reset_acc()

    def push(self, other):
        acc_base = unit_diff(self.pos2d, other.pos2d)
        acc_scale = overlapped_volume(self, other)
        other.acc2d += acc_base * (np.exp(acc_scale) - 1)

    def reset_acc(self):
        self.acc2d = np.zeros(2)

    def __str__(self):
        return f"{self.__class__.__name__}(Position={self.pos2d}, Velocity={self.vel2d}, Acceleration={self.acc2d})"
