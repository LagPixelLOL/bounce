import numpy as np
import multiprocessing
from .wall_object import WallType, WallObject

def calculate_acc(i, j, objects):
    acc_vec = objects[i].get_acc_to_target(objects[j])
    return (j, acc_vec)

class Box:

    def __init__(self, objects, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.objects = objects + [
            WallObject(WallType.HORIZONTAL_TOP, self.top_left[1]),
            WallObject(WallType.HORIZONTAL_BOTTOM, self.bottom_right[1]),
            WallObject(WallType.VERTICAL_LEFT, self.top_left[0]),
            WallObject(WallType.VERTICAL_RIGHT, self.bottom_right[0]),
        ]
        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    def step(self, t):
        objects_len = len(self.objects)
        tasks = [(i, j, self.objects) for i in range(objects_len) for j in range(objects_len) if i != j]
        acc_vecs = self.pool.starmap(calculate_acc, tasks)
        for i, acc_vec in acc_vecs:
            self.objects[i].accelerate(acc_vec)
        for object in self.objects:
            object.step(t)

    def render(self, resolution):
        horizontal_pixels = int((self.bottom_right[0] - self.top_left[0]) // resolution)
        vertical_pixels = int((self.top_left[1] - self.bottom_right[1]) // resolution)
        x_values = np.expand_dims(np.arange(horizontal_pixels).astype(np.float64), [0, -1]).repeat(vertical_pixels, 0) * resolution + self.top_left[0] + resolution / 2
        y_values = np.flip(np.expand_dims(np.arange(vertical_pixels).astype(np.float64), [0, -1]).repeat(horizontal_pixels, 0).transpose(1, 0, 2)) * resolution + self.bottom_right[1] + resolution / 2
        input_values = np.concat((x_values, y_values), axis=-1)
        objects = [object for object in self.objects if not isinstance(object, WallObject)]
        result = np.zeros([horizontal_pixels, vertical_pixels])
        for i in range(horizontal_pixels):
            for j in range(vertical_pixels):
                input = input_values[i][j]
                result[i][j] = max(object(input[0], input[1]) for object in objects)
        return np.clip(result, 0, 1)

    def __del__(self):
        self.pool.close()
        self.pool.join()
