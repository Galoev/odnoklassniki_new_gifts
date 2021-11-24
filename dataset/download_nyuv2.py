import numpy as np
import h5py
import cv2
from tqdm import tqdm
from pathlib import Path
from constants import PATH_TO_NYU


def read_db():
    # data path
    NYU_V2_FILE_NAME = 'nyu_depth_v2_labeled.mat'
    path_to_depth = PATH_TO_NYU/NYU_V2_FILE_NAME
    if not path_to_depth.is_file():
        raise FileNotFoundError(f"File: {NYU_V2_FILE_NAME} doesn't exist")

    # read mat file
    image_db = h5py.File(path_to_depth)
    print('Loaded NYU mat')
    return image_db
    

def extract_ds(image_db):
    # [3, 480, 640]
    data_dir = PATH_TO_NYU/'data'/'rgb'
    gt_dir = PATH_TO_NYU/'data'/'depth'

    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        print('Extracting RGB...')
        for i in tqdm(list(range(image_db['images'].shape[0]))):
            img = image_db['images'][i]
            img_ = np.empty([480, 640, 3])
            img_[:,:,0] = img[2,:,:].T
            img_[:,:,1] = img[1,:,:].T
            img_[:,:,2] = img[0,:,:].T
            cv2.imwrite(str(data_dir/(str(i) + '.jpg')), img_)
        print('Done')
    else:
        print("RGB Directory exists. Not extracting")
    
    if not gt_dir.exists():
        gt_dir.mkdir(parents=True)
        print('Extracting Normalized Depths...')
        for i in tqdm(list(range(image_db['depths'].shape[0]))):
            depth = image_db['depths'][i]
            my_depth = depth/4.0            
            my_depth_uint8 = cv2.normalize(src=my_depth.T, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.imwrite(str(gt_dir/(str(i) + '.png')), my_depth_uint8)
        print('Done')
    else:
        print("Depth Directory exists. Not extracting")

def show_example():
    rgb = PATH_TO_NYU/'data'/'rgb'/'1.jpg'
    depth = PATH_TO_NYU/'data'/'depth'/'1.png'
    im_rgb = cv2.imread(str(rgb))
    im_depth = cv2.imread(str(depth))
    
    cv2.imshow("RGB", im_rgb)
    cv2.imshow("Depth", im_depth)
    cv2.waitKey(0)


def run():
    image_db = read_db()
    extract_ds(image_db)


if __name__ == '__main__':
    run()
    show_example()
