import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from DataLocator import DataLocator

from Visualization import doVisualizaiton

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")

if __name__ == "__main__":

    print("\033[37m") # Light Gray
    
    dl_Test = DataLocator(TEST_DATA_FOLDER)
    dl_Train = DataLocator(TRAIN_DATA_FOLDER)

    test_files_json = dl_Test.get_files(".json")

    train_files_json = dl_Train.get_files(".json")

    print("\033[0m") # Reset Color

    
    doVisualizaiton(train_files_json,test_files_json)
    