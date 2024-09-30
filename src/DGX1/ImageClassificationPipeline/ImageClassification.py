import torch
import os

from time import sleep

from DataLocator import DataLocator
from Classifiaction import runClassification

from tqdm import tqdm

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")


def checkCuda():
    # Print PyTorch version
    print("PyTorch version: ", torch.__version__)

    # Check CUDA Version
    print("CUDA Version: ", torch.version.cuda)

    # Check if GPU is available
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    # Check and list available devices
    print("GPU Available devices: ", torch.cuda.device_count())

    # List all available devices
    for i in range(torch.cuda.device_count()):
        print(" -Device {}: {}".format(i, torch.cuda.get_device_name(i)))

    return "cuda" if torch.cuda.is_available() else "cpu"

def getRegionTypes():
    return {
    "PopulationAreas": [
        "City",
        "Suburbs",
        "Countryside"
    ],
    "Regions":[
        "Coastal",
        "Mountainous",
        "Desert",
        "Woodlands",
        "Industrial",
        "Agricultural",
        "Wetlands",
        "Highlands",
        "Urban"
    ]
}


if __name__ == '__main__':
    device = checkCuda()

    dl_Test = DataLocator(TEST_DATA_FOLDER)
    dl_Train = DataLocator(TRAIN_DATA_FOLDER)

    test_files_json = dl_Test.get_files(".json")
    train_files_json = dl_Train.get_files(".json")

    runClassification(train_files_json, getRegionTypes(), device)



    
    
    
