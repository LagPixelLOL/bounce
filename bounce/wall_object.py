import numpy as np
from enum import Enum
from .base_object import BaseObject

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
            raise ValueError(f"wall_type parameter must be of class {WallType.__name__}, instead got {wall_type.__class__.__name__}!")
        super().__init__(None, None, True)
        self.wall_type = wall_type
        self.line = line

    def __call__(self, x, y):
        match self.wall_type:
            case WallType.HORIZONTAL_TOP:
                return (y >= self.line).astype(np.float64)
            case WallType.HORIZONTAL_BOTTOM:
                return (y <= self.line).astype(np.float64)
            case WallType.VERTICAL_LEFT:
                return (x <= self.line).astype(np.float64)
            case WallType.VERTICAL_RIGHT:
                return (x >= self.line).astype(np.float64)
        raise NotImplementedError(f"Unknown wall_type {self.wall_type}!")

    def get_acc_to_target(self, target, acc_scale):
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
        return acc_base * (np.exp(acc_scale * 2) - 1)

    def __str__(self):
        return f"{self.__class__.__name__}(Wall type={self.wall_type}, Line={self.line})"
