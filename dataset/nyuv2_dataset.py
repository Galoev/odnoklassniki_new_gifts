
import numpy as np
from path import Path
import imageio as iio
from torch.utils.data import Dataset
from constants import PATH_TO_NYU


class NYUv2Dataset(Dataset):
    def __init__(self, root=PATH_TO_NYU, seed=None, train=True):
        np.random.seed(seed)
        self.root = Path(root)

    def __getitem__(self, index):
        rgb = iio.imread(self.root)
        depth = iio.imread(self.root)

        return rgb, depth
