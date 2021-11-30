import wget
import tarfile
from constants import RAW_NYU_V2_URL
from constants import RAW_NYU_V2_FILE_NAME
from constants import PATH_TO_NYU


class NYU2:
    def __init__(self):
        self.path_to_raw_nyu = PATH_TO_NYU/RAW_NYU_V2_FILE_NAME

    def download(self):
        self.__download_raw_nyu2()
        self.__extract_tar()

    def __download_raw_nyu2(self):
        if not self.path_to_raw_nyu.is_file():
            print("Start download raw dataset...")
            wget.download(RAW_NYU_V2_URL, str(self.path_to_raw_nyu))
            print("Downloading is complete")
        else:
            print(f"{RAW_NYU_V2_FILE_NAME} exist")

    def __extract_tar(self):
        if not self.path_to_raw_nyu.is_file():
            raise FileNotFoundError(f"{RAW_NYU_V2_FILE_NAME} doesn't exist")
        print(f"I start to extract the {RAW_NYU_V2_FILE_NAME}...")
        with tarfile.open(str(self.path_to_raw_nyu)) as tar:
            tar.extractall(PATH_TO_NYU)
        print(f"Extracting completed")

    def __move_all_h5_to_one_folder():
        pass


if __name__ == '__main__':
    dataset = NYU2()
    dataset.download()
