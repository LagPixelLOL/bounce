import numpy as np
from .utils import unit_diff, overlapped_volume

class BaseObject:

    def __init__(self, pos2d, vel2d=np.zeros(2), frozen=False):
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

    def get_acc_to_target(self, target, acc_scale):
        return None if target.frozen else unit_diff(self.pos2d, target.pos2d) * (np.exp(acc_scale * 2) - 1)

    # Returns (Self to other, Other to self).
    def get_mutual_acc(self, other):
        if self.frozen and other.frozen:
            return (None, None)
        acc_scale = overlapped_volume(self, other)
        return (self.get_acc_to_target(other, acc_scale), other.get_acc_to_target(self, acc_scale))

    def accelerate(self, acc_vec):
        if self.frozen or acc_vec is None:
            return
        self.acc2d += acc_vec

    def reset_acc(self):
        if self.frozen:
            return
        self.acc2d = np.zeros(2)

    def render(self, x_matrix, y_matrix):
        raise NotImplementedError

    def __str__(self):
        if self.frozen:
            return f"{self.__class__.__name__}(Position={self.pos2d}, Frozen=True)"
        return f"{self.__class__.__name__}(Position={self.pos2d}, Velocity={self.vel2d}, Acceleration={self.acc2d})"
