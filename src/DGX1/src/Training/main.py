import torch
import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')
sys.path.append('/Users/tobiasrothlin/Documents/MSE/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from tqdm import tqdm
from random import randint

import json

from GeoLocalizationDataset import GeoLocalizationDataset
from torch.utils.data import DataLoader
from Model import GeoLocalizationModel
from HaversineLoss import HaversineLoss

from torchsummary import summary

import matplotlib.pyplot as plt

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"
BASE_PATH = "/Users/tobiasrothlin/Documents/MSE/Dataset"

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


        model = GeoLocalizationModel(config["BaseModel"])

        summary(model, (3, config["ImageHeight"], config["ImageWidth"]))

        model.to(device)

        loss_function = HaversineLoss()


        # Train the model

        for epoch in range(config["Epochs"]):
            print(f"Epoch {epoch+1}/{config['Epochs']}")
            model.train()

            lat_loss = 0
            lon_loss = 0

            for i, (images, labels) in enumerate(tqdm(train_loader, desc="Training", postfix=f"Loss: {lat_loss:.2f},{lon_loss:.2f}")):
                images = images.to(device)
                labels = labels.to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=config["LearningRate"])

                optimizer.zero_grad()

                outputs = model(images)

                loss = loss_function(outputs, labels)
                
                loss.backward()

                optimizer.step()

            model.eval()
            with torch.no_grad():
                lat_loss = 0
                lon_loss = 0

                for i, (images, labels) in enumerate(tqdm(test_loader, desc="Testing",postfix=f"Loss: {lat_loss:.2f},{lon_loss:.2f}")):
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)

                    loss = loss_function(outputs, labels)

                    lat_loss += loss[0].item()
                    lon_loss += loss[1].item()


            # Save model Checkpoint
            torch.save(model.state_dict(), f"./model_{epoch}.pt")

            





                    
        