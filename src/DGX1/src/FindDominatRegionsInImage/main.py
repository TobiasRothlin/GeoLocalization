import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from DataLocator import DataLocator
from FindDominantRegionsInImage import findDominantRegionsInImagesbatched

from tqdm import tqdm

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")

if __name__ == "__main__":


    dl_Train = DataLocator(TRAIN_DATA_FOLDER)

    train_files_jpg = dl_Train.get_files(".jpg")
    train_files_jpg = sorted(train_files_jpg)


    train_files_jpg_by_folder = {}

    for file in train_files_jpg:
        folder = file.split("/")[-2]
        if folder not in train_files_jpg_by_folder:
            train_files_jpg_by_folder[folder] = []
        train_files_jpg_by_folder[folder].append(file)

    
    findDominantRegionsInImagesbatched(train_files_jpg_by_folder, num_threads=64)
        

    print(f"Done")

    