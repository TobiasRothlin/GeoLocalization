import torch
import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from tqdm import tqdm
from random import randint

import json

from Dataset import GeoLocalizationDataset
from torch.utils.data import DataLoader
from Model import GeoLocalizationModel

import matplotlib.pyplot as plt

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



if __name__ == '__main__':
    CHECK_IMAGE_FILES = True

    device = checkCuda()

    with open("./config.json", "r") as f:
        configs = json.load(f)

    for config in configs["Runs"]:
        # Create the dataset
        # train_dataset = GeoLocalizationDataset(TRAIN_DATA_FOLDER)
        test_dataset = GeoLocalizationDataset(TEST_DATA_FOLDER,
                                            image_width=config["ImageWidth"],
                                            image_height=config["ImageHeight"],
                                            use_center_crop=config["UseCenterCrop"],
                                            check_images=CHECK_IMAGE_FILES)
        
        train_dataset = GeoLocalizationDataset(TRAIN_DATA_FOLDER,
                                                image_width=config["ImageWidth"],
                                                image_height=config["ImageHeight"],
                                                use_center_crop=config["UseCenterCrop"],
                                                check_images=CHECK_IMAGE_FILES)
        
        # Create the dataloader
        train_loader = DataLoader(train_dataset, batch_size=config["TrainBatchSize"], shuffle=True,num_workers=config["NumWorkers"],persistent_workers=config["PersistantWorkers"], prefetch_factor=config["PrefetchFactor"])
        test_loader = DataLoader(test_dataset, batch_size=config["TestBatchSize"], shuffle=True,num_workers=config["NumWorkers"])


        model = GeoLocalizationModel(config["BaseModel"],device)

        output_list = []

        for batch in tqdm(train_loader, desc="Training"):
            image, label = batch
            device_image = image.to(device)

            output = model(device_image)
            
            
            

        


        




    
    

    




    
    
    
