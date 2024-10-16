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
from torch.utils.data.distributed import DistributedSampler
from Model import GeoLocalizationModel
from HaversineLoss import HaversineLoss
from Trainer import Trainer, ddp_setup

from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from torchsummary import summary

import matplotlib.pyplot as plt

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"
# BASE_PATH = "/Users/tobiasrothlin/Documents/MSE/Dataset"

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

def run(rank,world_size,config,test_dataset,train_dataset,model,loss_function):
    ddp_setup(rank, world_size)
    # Create the dataloader
    train_loader = DataLoader(train_dataset, batch_size=config["TrainingConfig"]["TrainBatchSize"], shuffle=False,num_workers=config["TrainingConfig"]["NumWorkers"],persistent_workers=config["TrainingConfig"]["PersistantWorkers"], prefetch_factor=config["TrainingConfig"]["PrefetchFactor"], sampler=DistributedSampler(train_dataset),pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["TrainingConfig"]["TestBatchSize"], shuffle=True,num_workers=config["TrainingConfig"]["NumWorkers"])

    # Train the model

    trainer = Trainer(model, 
                        train_loader, 
                        test_loader, 
                        "Adam", 
                        config["TrainingConfig"]["SaveEvery"], 
                        config["TrainingConfig"]["SnapshotPath"], 
                        loss_function, 
                        rank,
                        config["TrainingConfig"]["GradientAccumulationSteps"])
    

    trainer.train(config["TrainingConfig"]["Epochs"])

    destroy_process_group()


if __name__ == '__main__':
    

    device = checkCuda()

    with open("./config.json", "r") as f:
        configs = json.load(f)

    

    for config in configs["Runs"]:
        CHECK_IMAGE_FILES = True

        test_dataset = GeoLocalizationDataset(TEST_DATA_FOLDER,
                                            image_width=config["ModelConfig"]["ImageWidth"],
                                            image_height=config["ModelConfig"]["ImageHeight"],
                                            use_center_crop=config["ModelConfig"]["UseCenterCrop"],
                                            check_images=CHECK_IMAGE_FILES,
                                            image_mean=config["ModelConfig"]["ImageMean"],
                                            image_std=config["ModelConfig"]["ImageStd"])
        
        train_dataset = GeoLocalizationDataset(TRAIN_DATA_FOLDER,
                                            image_width=config["ModelConfig"]["ImageWidth"],
                                            image_height=config["ModelConfig"]["ImageHeight"],
                                            use_center_crop=config["ModelConfig"]["UseCenterCrop"],
                                            check_images=CHECK_IMAGE_FILES,
                                            image_mean=config["ModelConfig"]["ImageMean"],
                                            image_std=config["ModelConfig"]["ImageStd"])
        
        model = GeoLocalizationModel(config["ModelConfig"]["BaseModel"])

        summary(model, (3, config["ModelConfig"]["ImageHeight"], config["ModelConfig"]["ImageWidth"]))

        loss_function = HaversineLoss()

        # Create the dataset
        mp.spawn(run, args=(torch.cuda.device_count(),config,test_dataset,train_dataset,model,loss_function), nprocs=torch.cuda.device_count())


        

        



                    
        