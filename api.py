import os
import numpy as np

from predictor import PosPrediction


class PRN:
    ''' Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network
    Args:
        is_dlib(bool, optional): If true, dlib is used for detecting faces.
        is_opencv(bool, optional): If true, opencv is used for extracting texture.
        prefix(str, optional): If run at another folder, the absolute path is needed to load the data.
    '''
    def __init__(self, prefix = '.'):

        # resolution of input and output image size.
        self.resolution_inp = 256
        self.resolution_op = 256

        #---- load PRN 
        self.pos_predictor = PosPrediction(self.resolution_inp, self.resolution_op)
        prn_path = os.path.join(prefix, 'Data/net-data/256_256_resfcn256_weight')
        if not os.path.isfile(prn_path + '.data-00000-of-00001'):
            print("please download PRN trained model first.")
            exit()
        self.pos_predictor.restore(prn_path)

        # uv file

        self.face_ind = np.loadtxt(prefix + '/Data/uv-data/face_ind.txt').astype(np.int32) # get valid vertices in the pos map


    def net_forward(self, image):
        ''' The core of out method: regress the position map of a given image.
        Args:
            image: (256,256,3) array. value range: 0~1
        Returns:
            pos: the 3D position map. (256, 256, 3) array.
        '''
        return self.pos_predictor.predict(image)

    def get_vertices(self, pos):
        '''
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
        '''
        all_vertices = np.reshape(pos, [self.resolution_op**2, -1]);
        vertices = all_vertices[self.face_ind, :]

        return vertices
