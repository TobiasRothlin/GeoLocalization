import torch
import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')
sys.path.append('/Users/tobiasrothlin/Documents/MSE/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from tqdm import tqdm
from random import randint

import json

import dotenv
import mlflow

from GeoLocalizationDatasetDecoder import GeoLocalizationDatasetDecoder
from GaussianSmoothingScheduler import GaussianSmoothingScheduler
from Model import ClipLocationDecoder
from Trainer import Trainer
from HaversineLoss import HaversineLoss

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

from torchinfo import summary

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


if __name__ == '__main__':
    
    device = checkCuda()

    with open("./config.json", "r") as f:
        configs = json.load(f)

    


    for config in configs["Runs"]:
        gaussian_smoothing_scheduler = GaussianSmoothingScheduler(50,0.5,0.0)
        
        test_dataset = GeoLocalizationDatasetDecoder(TEST_DATA_FOLDER,
                 error_output="./error_output.txt",
                 standardization_coordinates=config["ModelConfig"]["StandardizationCoordinates"],
                 use_cache=True,
                 encoder_model=config["ModelConfig"]["BaseModel"])
        

        print(f"Test dataset size: {len(test_dataset.data)}")

        train_dataset = GeoLocalizationDatasetDecoder(TEST_DATA_FOLDER,
                 error_output="./error_output.txt",
                 standardization_coordinates=config["ModelConfig"]["StandardizationCoordinates"],
                 use_cache=True,
                 encoder_model=config["ModelConfig"]["BaseModel"],
                 std_dev_km=1,
                 std_dev_scheduler=gaussian_smoothing_scheduler)
        
        print(f"Train dataset size: {len(train_dataset.data)}")


        test_data_loader = DataLoader(test_dataset, 
                                        batch_size=config["TrainingConfig"]["TestBatchSize"], 
                                        shuffle=True,
                                        num_workers=config["TrainingConfig"]["NumWorkers"],
                                        persistent_workers=config["TrainingConfig"]["PersistantWorkers"], 
                                        prefetch_factor=config["TrainingConfig"]["PrefetchFactor"])
        
        train_data_loader = DataLoader(train_dataset, 
                                        batch_size=config["TrainingConfig"]["TrainBatchSize"], 
                                        shuffle=True,
                                        num_workers=config["TrainingConfig"]["NumWorkers"],
                                        persistent_workers=config["TrainingConfig"]["PersistantWorkers"], 
                                        prefetch_factor=config["TrainingConfig"]["PrefetchFactor"])
        

        model = ClipLocationDecoder()


        vector, label = train_dataset[0]
        summary(model, input_size=(1,vector.shape[0],vector.shape[1]))

        loss_function = HaversineLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=config["TrainingConfig"]["LearningRate"],
                                      amsgrad=config["TrainingConfig"]["Amsgrad"], 
                                      weight_decay=config["TrainingConfig"]["WeightDecay"], 
                                      betas=config["TrainingConfig"]["Betas"])

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["TrainingConfig"]["Gamma"])

        trainer = Trainer(train_dataloader=train_data_loader,
                            test_dataloader=test_data_loader,
                            model=model,
                            loss_function =loss_function ,
                            optimizer = optimizer,
                            lr_scheduler = lr_scheduler,
                            gaussian_smoothing_scheduler = gaussian_smoothing_scheduler,
                            gradient_accumulation_steps = config["TrainingConfig"]["GradientAccumulationSteps"],
                            epochs = config["TrainingConfig"]["Epochs"],
                            device = device,
                            log_interval= config["TrainingConfig"]["SaveEvery"],
                            snapshot_path= config["TrainingConfig"]["SnapshotPath"],
                            log_mlflow=False,
                            mlflow_experiment_name="GeoLocalization_Decoder_Model")
        

        trainer.train()


        




        

        



                    
        