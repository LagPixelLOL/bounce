import numpy as np
from .utils import unit_diff, overlapped_volume

class BaseObject:

    def __init__(self, pos2d, vel2d, frozen=False):
        self.pos2d = pos2d
        self.frozen = frozen
        if not self.frozen:
            self.vel2d = vel2d
            self.reset_acc()

    def __call__(self, x, y):
        raise NotImplementedError

    def step(self, t):
        if self.frozen:
            return
        self.vel2d += self.acc2d * t
        self.pos2d += self.vel2d * t
        self.reset_acc()

    def get_acc_to_target(self, target):
        if target.frozen:
            return None
        acc_base = unit_diff(self.pos2d, target.pos2d)
        acc_scale = overlapped_volume(self, target)
        return acc_base * (np.exp(acc_scale) - 1)

    def accelerate(self, acc_vec):
        if self.frozen or acc_vec is None:
            return
        self.acc2d += acc_vec

    def reset_acc(self):
        if self.frozen:
            return
        self.acc2d = np.zeros(2)

    def __str__(self):
        if self.frozen:
            return f"{self.__class__.__name__}(Position={self.pos2d}, Frozen=True)"
        return f"{self.__class__.__name__}(Position={self.pos2d}, Velocity={self.vel2d}, Acceleration={self.acc2d})"
