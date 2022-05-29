import numpy as np
import scipy.signal as signal
from scipy import interpolate


def z_normalize(z_v):
    eps = 1e-3
    return (z_v - np.min(z_v)) / (np.max(z_v) - np.min(z_v) + eps)


def convert_vertices_2_map_interp2d_directly(vertices, shape):
    image_a = np.zeros([shape[0], shape[1]], dtype=np.float32)
    z = 255 * z_normalize(vertices[:, 2])
    points = vertices[:, :2]
    x1, x2, y1, y2 = np.min(vertices[:, 0]), np.max(vertices[:, 0]), np.min(vertices[:, 1]), np.max(vertices[:, 1])
    x_new, y_new = np.meshgrid(np.linspace(x1, x2, int(x2 - x1)), np.linspace(y1, y2, int(y2 - y1)))
    z_new = interpolate.griddata(points, z, (x_new, y_new), method='linear', fill_value=0)
    z_new[z_new == np.nan] = 0
    x_cor = np.array(x_new, np.int32)
    y_cor = np.array(y_new, np.int32)
    image_index = x_cor.flatten() + y_cor.flatten() * shape[1]
    image_index[image_index >= shape[0] * shape[1]] = shape[0] * shape[1] - 1
    image_index = np.array(image_index, np.int32)
    image_a.ravel()[image_index] = z_new.flatten()

    return image_a


def generate_depth_image(vertices, shape, is_med_filter=False):
    vertices_map = convert_vertices_2_map_interp2d_directly(vertices, shape)

    if is_med_filter:
        vertices_map_med = signal.medfilt2d(vertices_map, (5, 5))
        vertices_map[vertices_map == 0] = vertices_map_med[vertices_map == 0]

    depth_map = np.array(vertices_map, np.uint8)

    return depth_map
