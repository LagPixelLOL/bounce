import numpy as np
import multiprocessing
from .wall_object import WallType, WallObject

def calculate_acc(i, j, objects):
    i_to_j, j_to_i = objects[i].get_mutual_acc(objects[j])
    return ((j, i_to_j), (i, j_to_i))

def object_render(object, x_matrix, y_matrix):
    return object.render(x_matrix, y_matrix)

class Box:

    def __init__(self, objects, top_left, bottom_right, color3d=np.zeros(3, dtype=np.uint8)):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.objects = objects + [
            WallObject(WallType.HORIZONTAL_TOP, self.top_left[1]),
            WallObject(WallType.HORIZONTAL_BOTTOM, self.bottom_right[1]),
            WallObject(WallType.VERTICAL_LEFT, self.top_left[0]),
            WallObject(WallType.VERTICAL_RIGHT, self.bottom_right[0]),
        ]
        self.color3d = color3d
        self.pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    def step(self, t):
        objects_len = len(self.objects)
        tasks = []
        for i in range(objects_len):
            for j in range(i + 1, objects_len):
                tasks.append((i, j, self.objects))
        acc_vecs = self.pool.starmap(calculate_acc, tasks)
        for i_to_j, j_to_i in acc_vecs:
            self.objects[i_to_j[0]].accelerate(i_to_j[1])
            self.objects[j_to_i[0]].accelerate(j_to_i[1])
        for object in self.objects:
            object.step(t)

    def make_render_input(self, resolution):
        horizontal_pixels = int((self.bottom_right[0] - self.top_left[0]) // resolution)
        vertical_pixels = int((self.top_left[1] - self.bottom_right[1]) // resolution)
        x_matrix = np.expand_dims(np.arange(horizontal_pixels).astype(np.float64), 0).repeat(vertical_pixels, 0) * resolution + self.top_left[0] + resolution / 2
        y_matrix = np.flip(np.expand_dims(np.arange(vertical_pixels).astype(np.float64), 0).repeat(horizontal_pixels, 0).transpose(1, 0)) * resolution + self.bottom_right[1] + resolution / 2
        return x_matrix, y_matrix

    def render(self, resolution):
        input_values = self.make_render_input(resolution)
        render_results = self.pool.starmap(object_render, [(object,) + input_values for object in self.objects if not isinstance(object, WallObject)])
        rendered_tensor = np.zeros(input_values[0].shape + (3,), dtype=np.uint32)
        weight_tensor = np.zeros(input_values[0].shape)
        for rendered_matrix, weight_matrix in render_results:
            rendered_tensor += rendered_matrix
            weight_tensor += weight_matrix
        weight_tensor = 1 - np.clip(weight_tensor, 0, 1)
        background_tensor = np.expand_dims(self.color3d, [0, 1]).repeat(input_values[0].shape[0], 0).repeat(input_values[0].shape[1], 1)
        return np.clip(rendered_tensor + np.round(background_tensor * np.expand_dims(weight_tensor, -1)).astype(np.uint8), 0, 255).astype(np.uint8)

    def __del__(self):
        self.pool.close()
        self.pool.join()
