import numpy as np
import h5py
import os
import cv2
from tqdm import tqdm
from constants import PATH_TO_NYU


def read_db():
    # data path
    path_to_depth = "".join([PATH_TO_NYU, '/nyu_depth_v2_labeled.mat'])

    # read mat file
    image_db = h5py.File(path_to_depth)
    print('Loaded NYU mat')
    return image_db
    

def extract_ds(image_db):
    # [3, 480, 640]
    data_dir = "".join([PATH_TO_NYU, '/data/rgb/'])
    gt_dir = "".join([PATH_TO_NYU, '/data/depth/'])

    if not(os.path.exists(data_dir)):
        os.makedirs(data_dir)
        print('Extracting RGB...')
        for i in tqdm(list(range(image_db['images'].shape[0]))):
            img = image_db['images'][i]
            img_ = np.empty([480, 640, 3])
            img_[:,:,0] = img[2,:,:].T
            img_[:,:,1] = img[1,:,:].T
            img_[:,:,2] = img[0,:,:].T
            cv2.imwrite("".join([data_dir, str(i), '.jpg']), img_)
        print('Done')
    else:
        print("RGB Directory exists. Not extracting")
    
    if not(os.path.exists(gt_dir)):
        os.makedirs(gt_dir)
        print('Extracting Normalized Depths...')
        for i in tqdm(list(range(image_db['depths'].shape[0]))):
            depth = image_db['depths'][i]
            my_depth = depth/4.0            
            my_depth_uint8 = cv2.normalize(src=my_depth.T, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.imwrite("".join([gt_dir, str(i), '.png']), my_depth_uint8)
        print('Done')
    else:
        print("Depth Directory exists. Not extracting")

def show_example():
    rgb = "".join([PATH_TO_NYU, '/data/rgb/1.jpg'])
    depth = "".join([PATH_TO_NYU, '/data/depth/1.png'])
    im_rgb = cv2.imread(rgb)
    im_depth = cv2.imread(depth)
    
    cv2.imshow("RGB", im_rgb)
    cv2.imshow("Depth", im_depth)
    cv2.waitKey(0)


def run():
    image_db = read_db()
    extract_ds(image_db)


if __name__ == '__main__':
    run()
    show_example()
