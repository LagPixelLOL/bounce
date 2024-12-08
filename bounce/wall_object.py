import numpy as np
from enum import Enum
from .base_object import BaseObject
from .utils import overlapped_volume

class WallType(Enum):
    HORIZONTAL_TOP = "Horizontal Top"
    HORIZONTAL_BOTTOM = "Horizontal Bottom"
    VERTICAL_LEFT = "Vertical Left"
    VERTICAL_RIGHT = "Vertical Right"

    def __init__(self, type_name):
        self.type_name = type_name

    def __str__(self):
        return self.type_name

class WallObject(BaseObject):

    def __init__(self, wall_type, line):
        if not isinstance(wall_type, WallType):
            raise ValueError(f"wall_type parameter must be class {WallType.__name__}, instead got {wall_type.__class__.__name__}!")
        super().__init__(None, None, True)
        self.wall_type = wall_type
        self.line = line

    def __call__(self, x, y):
        match self.wall_type:
            case WallType.HORIZONTAL_TOP:
                return int(y >= self.line)
            case WallType.HORIZONTAL_BOTTOM:
                return int(y <= self.line)
            case WallType.VERTICAL_LEFT:
                return int(x <= self.line)
            case WallType.VERTICAL_RIGHT:
                return int(x >= self.line)
        raise NotImplementedError(f"Unknown wall_type {self.wall_type}!")

    def get_acc_to_target(self, target):
        if target.frozen:
            return None
        match self.wall_type:
            case WallType.HORIZONTAL_TOP:
                acc_base = np.array([0.0, -1.0])
            case WallType.HORIZONTAL_BOTTOM:
                acc_base = np.array([0.0, 1.0])
            case WallType.VERTICAL_LEFT:
                acc_base = np.array([1.0, 0.0])
            case WallType.VERTICAL_RIGHT:
                acc_base = np.array([-1.0, 0.0])
            case _:
                raise NotImplementedError(f"Unknown wall_type {self.wall_type}!")
        acc_scale = overlapped_volume(self, target)
        return acc_base * (np.exp(acc_scale) - 1)

    def __str__(self):
        return f"{self.__class__.__name__}(Wall type={self.wall_type}, Line={self.line})"
