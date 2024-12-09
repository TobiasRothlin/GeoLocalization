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
from CosignSimilarityLoss import CosignSimilarityLoss
from EuclidianDistanceLoss import EuclidianDistanceLoss
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


def get_latest_model_checkpoint(path):
    files = os.listdir(path)
    files = [f for f in files if f.endswith(".pt")]
    files = sorted(files, key=lambda x: (int(x.split("_")[1])+1)*int(x.split("_")[3].replace(".pt","")))
    return os.path.join(path,files[-1])



if __name__ == '__main__':
    
    device = checkCuda()

    config_file = "./config_Regression_3.json"

    print(f"Using Config File: {config_file}")

    with open(config_file, "r") as f:
        configs = json.load(f)

    for config in configs["Runs"]:
        try:
            if "ResumeFromPreviousRun" in config:
                print("Resuming from previous run")
                did_resume = True
                previous_run_path = config["ResumeFromPreviousRun"]
                with open(previous_run_path+"/run_config.json", "r") as f:
                    config = json.load(f)

            else:
                did_resume = False
                previous_run_path = None

            dataset_test = GeoLocalizationDataset(TEST_DATA_FOLDER,**config["DatasetConfig"])
            dataset_train = GeoLocalizationDataset(TRAIN_DATA_FOLDER,**config["DatasetConfig"])

            print(f"Training Dataset: {len(dataset_train)}")
            print(f"Test Dataset: {len(dataset_test)}")
                
            model = LocationDecoder(config["ModelConfig"],
                                    base_model=config["DatasetConfig"]["base_model"],
                                    use_pre_calculated_embeddings=config["DatasetConfig"]["use_pre_calculated_embeddings"],
                                    freeze_base_model=config["ModelConfig"]["freeze_base_model"],)
            if did_resume:
                model.load(get_latest_model_checkpoint(previous_run_path))

            model.summary()

            if config["TrainingConfig"].get("ContrastLearningStrategy",None):
                if config["TrainingConfig"]["ContrastLearningStrategy"] == "CosignSimilarityLoss":
                    print("Using Cosign Similarity Loss")
                    loss_function = CosignSimilarityLoss()
                elif config["TrainingConfig"]["ContrastLearningStrategy"] == "EuclidianDistanceLoss":
                    print("Using Euclidian Distance Loss")
                    loss_function = EuclidianDistanceLoss()
                else:
                    raise ValueError("ContrastLearningStrategy not recognized")
            else:
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

        



        
        

        



                    
        