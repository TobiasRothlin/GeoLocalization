import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from DataLocator import DataLocator

from tqdm import tqdm

import shutil
import random

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")

DESTINATION_FOLDER = "/home/tobias.rothlin/data/GeoDataset/Test/GeoDataset2024"

HOLDOUT_DATA_SAMPELS = 5000

if __name__ == "__main__":

    dl_Train = DataLocator(TRAIN_DATA_FOLDER, use_cache=False)

    train_files_json = dl_Train.get_files(".json")
    train_files_jpg = dl_Train.get_files(".jpg")



    print(f"Creating holdout dataset")


    holdout_files_josn = random.sample(train_files_json, HOLDOUT_DATA_SAMPELS)


    for json_file in tqdm(holdout_files_josn, desc="Moving json files"):
        image_path = json_file.replace(".json", ".jpg")

        if not os.path.exists(image_path):
            image_path = json_file.replace(".json", ".jpeg")


        if os.path.exists(json_file) and os.path.exists(image_path):
            shutil.move(json_file, DESTINATION_FOLDER)
            shutil.move(image_path, DESTINATION_FOLDER)
        else:
            if not os.path.exists(json_file):
                print(f"File not found: {json_file}")
                raise FileNotFoundError(f"File not found: {json_file}")

            if not os.path.exists(image_path):
                print(f"File not found: {image_path}")
                raise FileNotFoundError(f"File not found: {image_path}")
            
        

            


   
    