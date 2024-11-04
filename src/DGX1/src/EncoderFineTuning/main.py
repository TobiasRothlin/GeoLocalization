import torch
import os


from time import sleep

from tqdm import tqdm
from random import randint

import json

import dotenv
import mlflow

from GeoLocalizationDataset import GeoLocalizationDataset


from Model import LocationDecoder
from HaversineLoss import HaversineLoss
from SingleGPUTrainer import SingleGPUTrainer
from MultiGPUTrainer import MultiGPUTraining

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
        dataset_test = GeoLocalizationDataset(TEST_DATA_FOLDER,**config["DatasetConfig"])
        dataset_train = GeoLocalizationDataset(TEST_DATA_FOLDER,**config["DatasetConfig"])
            
        model = LocationDecoder(config["ModelConfig"],
                                base_model=config["DatasetConfig"]["base_model"],
                                use_pre_calculated_embeddings=config["DatasetConfig"]["use_pre_calculated_embeddings"])
        
        
        model.summary()

        loss_function = HaversineLoss(use_standarized_input=config["DatasetConfig"]["normalize_labels"])

        optimizer = torch.optim.Adam(model.get_parameters(), 
                                     lr=config["TrainingConfig"]["LearningRate"],
                                     weight_decay=config["TrainingConfig"]["WeightDecay"],
                                     amsgrad=config["TrainingConfig"]["Amsgrad"],
                                     betas=config["TrainingConfig"]["Betas"])
        
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["TrainingConfig"]["Gamma"])
        
        if config["DataLoaderConfig"]["Is_Multi_GPU"]:
            MultiGPUTraining(config,
                            train_dataset=dataset_train,
                            test_dataset=dataset_test,
                            model=model,
                            loss_function=loss_function,
                            optimizer=None,
                            lr_scheduler=None)
        else:
            trainer = SingleGPUTrainer(test_dataset=dataset_test,
                                    train_dataset=dataset_train,
                                    test_dataloader_config=config["DataLoaderConfig"]["Test"],
                                    train_dataloader_config=config["DataLoaderConfig"]["Train"],
                                    model=model,
                                    loss_function=loss_function,
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler,
                                    gradient_accumulation_steps=config["TrainingConfig"]["GradientAccumulationSteps"],
                                    epochs=config["TrainingConfig"]["Epochs"],
                                    device=device,
                                    log_interval=config["TrainingConfig"]["SaveEvery"],
                                    snapshot_path=config["TrainingConfig"]["SnapshotPath"],
                                    log_mlflow=config["TrainingConfig"]["LogMLFlow"],
                                    mlflow_experiment_name=config["TrainingConfig"]["MLFlowExperimentName"],
                                    full_run_config=config)
            
            trainer.train()

        



        
        

        



                    
        