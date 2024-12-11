import torch
import os

from time import sleep

from tqdm import tqdm
from random import randint

import json

import dotenv
import mlflow

import traceback

from GeoLocalizationDataset import GeoLocalizationDataset

from Model import LocationDecoder
from HaversineLoss import HaversineLoss

from SingleGPUTrainer import SingleGPUTrainer

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


def get_latest_model_checkpoint(path):
    files = os.listdir(path)
    files = [f for f in files if f.endswith(".pt")]
    files = sorted(files, key=lambda x: (int(x.split("_")[1])+1)*int(x.split("_")[3].replace(".pt","")))
    return os.path.join(path,files[-1])



if __name__ == '__main__':
    
    device = checkCuda()

    config_file = "./config_Regression_Pretrained_7.json"

    print(f"Using Config File: {config_file}")

    with open(config_file, "r") as f:
        configs = json.load(f)

    for config in configs["Runs"]:
        try:
            dataset_test = GeoLocalizationDataset(TEST_DATA_FOLDER,**config["DatasetConfig"])
            dataset_train = GeoLocalizationDataset(TRAIN_DATA_FOLDER,**config["DatasetConfig"])

            print(f"Training Dataset: {len(dataset_train)}")
            print(f"Test Dataset: {len(dataset_test)}")

            if "Pretraining_weights" and "Pretraining_config" in config:
                with open(config["Pretraining_config"], "r") as f:
                    pretraining_config = json.load(f)
                model = LocationDecoder(pretraining_config["ModelConfig"],
                                        base_model=pretraining_config["DatasetConfig"]["base_model"],
                                        use_pre_calculated_embeddings=pretraining_config["DatasetConfig"]["use_pre_calculated_embeddings"],
                                        freeze_base_model=pretraining_config["ModelConfig"]["freeze_base_model"],)
                model.load(config["Pretraining_weights"])
            
                model.summary()

                model.reconfigure(config["ModelConfig"])

                model.summary()
            else:    
                raise ValueError("Pretraining weights and config not found in config file")


            loss_function = HaversineLoss(use_standarized_input=config["DatasetConfig"]["normalize_labels"])

            optimizer = torch.optim.Adam(model.parameters(), 
                                        lr=config["TrainingConfig"]["LearningRate"],
                                        weight_decay=config["TrainingConfig"]["WeightDecay"],
                                        amsgrad=config["TrainingConfig"]["Amsgrad"],
                                        betas=config["TrainingConfig"]["Betas"])
            
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["TrainingConfig"]["Gamma"])
        
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
                                        full_run_config=config,
                                        contrast_learning_strategy = config["TrainingConfig"].get("ContrastLearningStrategy",None),
                                        run_name=config["TrainingConfig"].get("RunName",None))
            
            trainer.train()

        except Exception as e:
            run_name = config["TrainingConfig"].get("RunName","NoName")
            with open(f"Error_{run_name}.txt", "w") as f:
                f.write(f"Error in run: {config}\n")
                f.write(str(e))
                f.write(traceback.format_exc())
                f.write("\n")

            print(f"Error in run: {config}")
            print(e)
            print(traceback.format_exc())

        



        
        

        



                    
        