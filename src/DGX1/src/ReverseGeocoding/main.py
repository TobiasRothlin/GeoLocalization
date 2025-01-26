import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from DataLocator import DataLocator
from ReverseGeocoding import put_locations_to_json_files

from tqdm import tqdm

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")

if __name__ == "__main__":

    print("\033[37m") # Light Gray
    
    dl_Test = DataLocator(TEST_DATA_FOLDER,use_cache=False)
    dl_Train = DataLocator(TRAIN_DATA_FOLDER,use_cache=False)

    test_files_json = dl_Test.get_files(".json")
    test_files_jpg = dl_Test.get_files(".jpg")
    test_files_jpeg = dl_Test.get_files(".jpeg")

    print(len(test_files_json))

    train_files_json = dl_Train.get_files(".json")
    train_files_jpg = dl_Train.get_files(".jpg")
    train_files_jpeg = dl_Train.get_files(".jpeg")

    print("\033[0m") # Reset Color

    print(f"Reverse Geo Coding Test Data")
    put_locations_to_json_files(test_files_json, num_threads=64, error_log="error_log_test.txt")

    print(f"Reverse Geo Coding Train Data")
    put_locations_to_json_files(train_files_json, num_threads=64, error_log="error_log_train.txt")

    print(f"Done")

    