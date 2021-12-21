import h5py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DatasetNYUv2(Dataset):
    PATH_TO_NYU = Path("/Volumes/TMP_Storage/Datasets/NYUv2/datasets.lids.mit.edu")

    def __init__(self, path_to_dataset: Path=PATH_TO_NYU, seed: int=42, train: bool=True):
        np.random.seed(seed)
        
        self.train = train 
        self.path_to_nyu_dir = path_to_dataset

        TRAIN_DIR_NAME = 'train'
        VAL_DIR_NAME = 'val'

        if self.train:
            self.path_to_data = self.path_to_nyu_dir / TRAIN_DIR_NAME            
        else:
            self.path_to_data = self.path_to_nyu_dir / VAL_DIR_NAME

        self.paths_to_data = list(self.path_to_data.glob("*.h5"))
        self.data_len = len(self.paths_to_data)

    def __getitem__(self, index):
        path_to_data = self.paths_to_data[index]
        h5_data = h5py.File(path_to_data)

        rgb_img = np.array(h5_data['rgb'][:])
        # rgb_img = rgb_img.transpose((1, 2, 0))
        rgb_img = torch.tensor(rgb_img)
        rgb_img = rgb_img.float()
        
        depth_img = np.array(h5_data['depth'][:])
        depth_img = torch.tensor(depth_img)
        depth_img = depth_img.float()
        depth_img = torch.clamp(depth_img, min=0, max=1)
        # depth_img /= 10
        # depth_img *= 255
        
        return rgb_img, depth_img

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    test_data = DatasetNYUv2(train=False)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)
    test_features, test_labels = next(iter(test_dataloader))
    rgb_img, depth_img = test_features[0], test_labels[0]

    rgb_img = rgb_img.numpy()
    depth_img = depth_img.numpy()

    pil_rgb_img = Image.fromarray(rgb_img, mode="RGB")
    pil_rgb_img.show("RGB")

    pil_depth_img = Image.fromarray(depth_img)
    pil_depth_img.show("depth")