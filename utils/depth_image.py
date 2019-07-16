import numpy as np
import os
import glob
import scipy.io as sio
from PIL import Image
import cv2
from scipy import interpolate  
import scipy.signal as signal

def crop_face(mat, kpt, scale, ori_shape, resize_shape):
    x1, x2, y1, y2 = kpt[:]

    x_mid = (x1 + x2)/2.0
    y_mid = (y1 + y2)/2.0
    h_half = (x2 - x1)/2.0
    w_half = (y2 - y1)/2.0

    x1 = x_mid - scale * h_half
    x2 = x_mid + scale * h_half
    y1 = y_mid - scale * w_half
    y2 = y_mid + scale * w_half

    x1 = max(x1, 0)
    x2 = min(x2, ori_shape[0])
    y1 = max(y1, 0)
    y2 = min(y2, ori_shape[1])

    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

    if len(mat.shape) == 3:
        crop_face_mat =  mat[y1:y2, x1:x2, :]
    else:
        crop_face_mat =  mat[y1:y2, x1:x2]

    crop_face_mat = cv2.resize(crop_face_mat, tuple(resize_shape),interpolation=cv2.INTER_CUBIC)
    return crop_face_mat

def z_normalize_partition(z_v):
    eps = 1e-3
    z_pos = np.zeros(z_v.shape)
    z_neg = np.zeros(z_v.shape)
    if len(z_v[z_v > 0]) == 0 or len(z_v[z_v < 0]) == 0:
        return z_normalize(z_v)
    else:
        z_pos[z_v > 0] = 0.5 * z_normalize(z_v[z_v > 0]) + 0.5
        z_neg[z_v < 0] = 0.5 * z_normalize(np.abs(z_v[z_v < 0]))
        return z_pos + z_neg

def z_normalize(z_v):
    eps = 1e-3
    return (z_v - np.min(z_v))/(np.max(z_v)-np.min(z_v)+eps)

def convert_vertices_2_map_interp2d_directly(vertices, shape):
    image_a = np.zeros([shape[0], shape[1]], dtype = np.float32)

    image_index_ori = np.array(vertices, np.float32)

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = 255 * z_normalize(vertices[:, 2])
    points = vertices[:, :2]
    
    x1, x2, y1, y2 = np.min(vertices[:,0]), np.max(vertices[:,0]), np.min(vertices[:,1]), np.max(vertices[:,1])
    xnew, ynew = np.meshgrid(np.linspace(x1, x2, int(x2-x1)), np.linspace(y1, y2, int(y2-y1))) 
    #ynew = ynew.flatten()
    #xnew = xnew.flatten()
    
    znew = interpolate.griddata(points, z, (xnew, ynew), method='linear', fill_value = 0)
    znew[znew == np.nan] = 0
    
    x_cor = np.array(xnew, np.int32)
    y_cor = np.array(ynew, np.int32)
    image_index = x_cor.flatten() + y_cor.flatten() * shape[1]
    image_index[image_index>=shape[0]*shape[1]] = shape[0]*shape[1]-1
    image_index = np.array(image_index, np.int32)
    #print(image_index)
    image_a.ravel()[image_index] = znew.flatten()

    return image_a

def generate_depth_image(vertices, kpt, shape, isMedFilter=False):
    vertices_map = convert_vertices_2_map_interp2d_directly(vertices, shape)

    if isMedFilter:
        vertices_map_med = signal.medfilt2d(vertices_map, (5,5))
        vertices_map[vertices_map==0] = vertices_map_med[vertices_map==0]

    depth_map = np.array(vertices_map, np.uint8)

    return depth_map