import cv2
import time
import numpy as np
from glob import glob
from tqdm import tqdm
from skimage.transform import resize
from multiprocessing import Pool, cpu_count

from api import PRN
from config.pipeline import *
import utils.depth_image as DepthImage


prn = PRN()


def main():
    datasets_list = glob(f'{datasets_dir}/*.npy')

    for dataset in tqdm(datasets_list):
        ds = np.load(dataset, allow_pickle=True)
        labels = ds[:10, 0]
        datas = ds[:10, 1]
        idx = list(range(len(labels)))


        # MultiProcessing
        pool = Pool(processes=cpu_count())
        X = pool.starmap(build_map, zip(labels, datas, idx))
        pool.close()
        pool.join()
        save_file_name = dataset.split("\\")[-1].split(".")[0]
        np.save(f'{save_path}/{save_file_name}_W_depth.npy', np.array(X, dtype=object))


def build_map(label, image, idx):
    if label == 0:
        depth = np.zeros((32, 32))
    else:
        pos = prn.net_forward(image / 255.)
        # 3D vertices
        vertices = prn.get_vertices(pos)
        depth_scene_map = DepthImage.generate_depth_image(vertices, image.shape, is_med_filter=True)
        depth = resize(depth_scene_map,
                       output_shape=(32, 32),
                       preserve_range=True)

    cv2.imwrite(f'predict/{idx}.jpg', depth)

    return [label, image, depth]


if __name__ == '__main__':
    start = time.perf_counter()
    main()

    end = time.perf_counter()
    print('*' * 50)
    print(round(end - start))