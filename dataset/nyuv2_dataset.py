import h5py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional

output_height=256
output_width=256

class DatasetNYUv2(Dataset):
    PATH_TO_NYU = Path("/Volumes/TMP_Storage/Datasets/NYUv2/datasets.lids.mit.edu/_nyudepthv2")

    def __init__(self, path_to_dataset: Path=PATH_TO_NYU, seed: int=42, train: bool=True):
        np.random.seed(seed)
        
        self.train = train 
        self.path_to_nyu_dir = Path(path_to_dataset)

        TRAIN_DIR_NAME = 'train'
        VAL_DIR_NAME = 'val'

        if self.train:
            self.path_to_data = self.path_to_nyu_dir / TRAIN_DIR_NAME            
        else:
            self.path_to_data = self.path_to_nyu_dir / VAL_DIR_NAME

        self.paths_to_h5 = []
        self.__init_paths_to_h5()
        self.data_len = len(self.paths_to_h5)
        self.resize = T.Resize((output_height, output_width))
        # self.paths_to_data = list(self.path_to_data.glob("*.h5"))
        # self.data_len = len(self.paths_to_data)

    def __getitem__(self, index):
        path_to_data = self.paths_to_h5[index]
        h5_data = h5py.File(path_to_data)

        rgb_img = np.array(h5_data['rgb'][:])
        # rgb_img = rgb_img.transpose((1, 2, 0))
        rgb_img = torch.tensor(rgb_img)
        rgb_img = rgb_img.float()
        rgb_img = functional.resize(rgb_img, size=(output_height, output_width))
        
        depth_img = np.array(h5_data['depth'][:])
        depth_img = np.reshape(depth_img, (1, depth_img.shape[0], depth_img.shape[1]))
        depth_img = torch.tensor(depth_img)
        depth_img = depth_img.float()
        depth_img = torch.clamp(depth_img, min=0, max=1)
        depth_img = functional.resize(depth_img, size=(output_height, output_width))
        
        # depth_img /= 10
        # depth_img *= 255
        
        return rgb_img, depth_img

    def __len__(self):
        return self.data_len

    def __init_paths_to_h5(self):
        for dir in self.path_to_data.glob("*"):
            if dir.is_dir():
                for file in dir.glob('*.h5'):
                    self.paths_to_h5.append(file)


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