import wget
import tarfile
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

RAW_NYU_V2_URL = "http://datasets.lids.mit.edu/fastdepth/data/nyudepthv2.tar.gz"
PATH_TO_NYU = Path("/Volumes/TMP_Storage/Datasets/NYUv2/datasets.lids.mit.edu")
RAW_NYU_V2_FILE_NAME = "nyudepthv2.tar.gz"

class NYU2:
    def __init__(self, path_to_nyu=PATH_TO_NYU, raw_nyu_archive_name = "nyudepthv2.tar.gz"):
        self.path_to_raw_nyu_archive = path_to_nyu/raw_nyu_archive_name
        self.nyudepthv2 = RAW_NYU_V2_FILE_NAME.split(sep='.')[0]
        self.path_to_nyu_dir = path_to_nyu/self.nyudepthv2

    def download(self):
        if self.__download_raw_nyu2():
            self.__extract_tar()
        self.__move_all_h5_to_one_dir()

    def __download_raw_nyu2(self) -> bool:
        if self.path_to_nyu_dir.is_dir():
            print(f"The {self.nyudepthv2} directory exist. Dataset already downloaded and extracted.")
            return False

        if not self.path_to_raw_nyu_archive.is_file():
            print("Start download raw dataset...")
            wget.download(RAW_NYU_V2_URL, str(self.path_to_raw_nyu_archive))
            print("Downloading is complete")
        else:
            print(f"{RAW_NYU_V2_FILE_NAME} file exist. Therefore, the download is skipped.")
        
        return True

    def __extract_tar(self):
        if not self.path_to_raw_nyu_archive.is_file():
            raise FileNotFoundError(f"{RAW_NYU_V2_FILE_NAME} doesn't exist")
        
        if self.path_to_nyu_dir.is_dir():
            print(f"The {self.nyudepthv2} directory exist. Therefore, the extraction is skipped.")

        print(f"I start to extract the {RAW_NYU_V2_FILE_NAME}...")
        with tarfile.open(str(self.path_to_raw_nyu_archive)) as tar:
            tar.extractall(PATH_TO_NYU)
        print(f"Extracting completed")

    def __move_all_h5_to_one_dir(self):
        TRAIN_DIR_NAME = 'train'
        VAL_DIR_NAME = 'val'

        self.path_to_nyu_train_dir = self.path_to_nyu_dir / TRAIN_DIR_NAME
        self.path_to_nyu_val_dir = self.path_to_nyu_dir / VAL_DIR_NAME

        dirs = [self.path_to_nyu_train_dir, self.path_to_nyu_val_dir]

        for dir in dirs:
            if not self.__check_dir_for_h5(dir):
                print(f"Move all h5 files to the directory: {dir}")
                print(f"Start...")
                self.__move_for_one_dir(dir)
                print(f"End")
            else:
                print(f"Files from the directory {dir} have already been moved")

    def __move_for_one_dir(self, path_to_dir: Path):
        for dir in tqdm(path_to_dir.glob("*"), total=280):
            if dir.is_dir():
                self.__move_higher(dir)
    
    def __check_dir_for_h5(self, path_to_dir: Path):
        try:
            tmp = next(path_to_dir.glob("*.h5"))
            return True
        except StopIteration:
            return False

    def __move_higher(self, path_to_dir: Path):
        target = path_to_dir.parent
        path_name = path_to_dir.name

        for file in path_to_dir.glob('*.h5'):
            file.rename(target / f"{path_name}_{file.name}")
            
def parse_args() -> Namespace:
    parser = ArgumentParser(description='NYUv2 downloader')
    parser.add_argument('--data', type=str, default=PATH_TO_NYU, metavar='D',
                     help="folder where NYUv2 is located")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = NYU2(path_to_nyu=Path(args.data))
    dataset.download()
