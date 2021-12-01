import h5py
import numpy as np
from PIL import Image
from pathlib import Path
from constants import PATH_TO_NYU

def visualize_nyuv2_h5(path_to_data: Path):
    if not path_to_data.is_file():
        print(f"{path_to_data} is not file")
    
    h5_data = h5py.File(path_to_data)
    
    rgb_img = np.array(h5_data['rgb'][:])
    rgb_img = rgb_img.transpose((1, 2, 0))    
    pil_rgb_img = Image.fromarray(rgb_img, mode="RGB")
    pil_rgb_img.show("RGB")

    
    depth_img = np.array(h5_data['depth'][:])
    depth_img /= 10
    depth_img *= 255
    pil_depth_img = Image.fromarray(depth_img)
    pil_depth_img.show("depth")


if __name__ == '__main__':
    visualize_nyuv2_h5(PATH_TO_NYU / 'nyudepthv2' / 'val' / 'official' / '00028.h5')