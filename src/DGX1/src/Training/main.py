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
from VisualizeModelEmbeddings import visualize_model_embeddings

from transformers import CLIPConfig, CLIPVisionConfig, CLIPProcessor

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
    CHECK_IMAGE_FILES = False

    device = checkCuda()

    with open("./config.json", "r") as f:
        configs = json.load(f)

    for config in configs["Runs"]:
        # Create the dataset

        test_dataset = GeoLocalizationDataset(TEST_DATA_FOLDER,
                                            image_width=config["ImageWidth"],
                                            image_height=config["ImageHeight"],
                                            use_center_crop=config["UseCenterCrop"],
                                            check_images=CHECK_IMAGE_FILES,
                                            image_mean=config["ImageMean"],
                                            image_std=config["ImageStd"])
        
        train_dataset = GeoLocalizationDataset(TRAIN_DATA_FOLDER,
                                               image_width=config["ImageWidth"],
                                               image_height=config["ImageHeight"],
                                               use_center_crop=config["UseCenterCrop"],
                                               check_images=CHECK_IMAGE_FILES,
                                               image_mean=config["ImageMean"],
                                               image_std=config["ImageStd"])
        
        # Create the dataloader
        train_loader = DataLoader(train_dataset, batch_size=config["TrainBatchSize"], shuffle=True,num_workers=config["NumWorkers"],persistent_workers=config["PersistantWorkers"], prefetch_factor=config["PrefetchFactor"])
        test_loader = DataLoader(test_dataset, batch_size=config["TestBatchSize"], shuffle=True,num_workers=config["NumWorkers"])


        model = GeoLocalizationModel(config["BaseModel"],device)

        visualize_model_embeddings(model, train_dataset, device,config["RunName"]+"_train",number_of_samples=100_000)
        
            
            
            

        


        




    
    

    




    
    
    
