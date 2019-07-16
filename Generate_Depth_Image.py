import numpy as np
import scipy.io as sio
from skimage.io import imread, imsave
import cv2
import os

from api import PRN
import utils.depth_image as DepthImage

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

prn = PRN(is_dlib = True, is_opencv = False) 

path_image = './TestImages/0.jpg'

image = imread(path_image)
image_shape = [image.shape[0], image.shape[1]]

pos = prn.process(image, None, None, image_shape)

kpt = prn.get_landmarks(pos)

# 3D vertices
vertices = prn.get_vertices(pos)

depth_scene_map = DepthImage.generate_depth_image(vertices, kpt, image.shape, isMedFilter=True)

cv2.imshow('IMAGE', image[:,:,::-1])
cv2.imshow('DEPTH', depth_scene_map)
cv2.waitKey(3000)
