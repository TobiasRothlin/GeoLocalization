import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from DataLocator import DataLocator
from FindDominantRegionsInImage import findDominantRegionsInImages

from tqdm import tqdm

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")

if __name__ == "__main__":


    dl_Train = DataLocator(TRAIN_DATA_FOLDER+"/Batch_1546")

    train_files_jpg = dl_Train.get_files(".jpg")
    train_files_jpg = sorted(train_files_jpg)


    print(f"Found {len(train_files_jpg)} files")

    pil_image = findDominantRegionsInImages(train_files_jpg)
        
    pil_image.save("./mean_image.jpg")

    print(f"Done")

    