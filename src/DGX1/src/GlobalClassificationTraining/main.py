import torch
import os

from time import sleep

from tqdm import tqdm
from random import randint

import json

import dotenv
import mlflow

import traceback

from GeoLocalizationDatasetClassification import GeoLocalizationClassificationDataset

from Model import LocationDecoder
from HaversineLoss import HaversineLoss
from CosignSimilarityLoss import CosignSimilarityLoss
from EuclidianDistanceLoss import EuclidianDistanceLoss
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



if __name__ == '__main__':
    
    device = checkCuda()

    config_file = "./config_Classification_Country_Oceania.json"

    label_dict_train_path = "./label_dict_train.json"
    label_dict_test_path = "./label_dict_test.json"

    print(f"Using Config File: {config_file}")

    with open(config_file, "r") as f:
        configs = json.load(f)

    for config in configs["Runs"]:
        try:
            dataset_train = GeoLocalizationClassificationDataset(TRAIN_DATA_FOLDER,**config["DatasetConfig"])
            dataset_train.save_label_dict(label_dict_train_path)

            dataset_test = GeoLocalizationClassificationDataset(TEST_DATA_FOLDER,**config["DatasetConfig"])
            dataset_test.set_label_dict(dataset_train.get_label_dict())
            dataset_test.save_label_dict(label_dict_test_path)

            if "UseClassWeightsCrossEntropy" in config["TrainingConfig"] and config["TrainingConfig"]["UseClassWeightsCrossEntropy"]:
                print("Using Class Weights")
                class_weights = dataset_train.get_label_histogram()
                class_weights = class_weights.to(device)
                print(class_weights)
            else:
                class_weights = None
            
            print(f"Training Dataset: {len(dataset_train)}")
            print(f"Test Dataset: {len(dataset_test)}")

            dataloader_train = torch.utils.data.DataLoader(dataset_train, **config["DataLoaderConfig"]["Train"])
            dataloader_test = torch.utils.data.DataLoader(dataset_test, **config["DataLoaderConfig"]["Test"])

            batch = next(iter(dataloader_train))
            
            image, label = batch

            print(f"Image Shape: {image.shape}")
            print(f"Label Shape: {label.shape}")

            model = LocationDecoder(config["ModelConfig"],
                                    base_model=config["DatasetConfig"]["base_model"],
                                    use_pre_calculated_embeddings=config["DatasetConfig"]["use_pre_calculated_embeddings"],
                                    freeze_base_model=config["ModelConfig"]["freeze_base_model"],)
            
            model.summary()
            

            loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

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
                                        run_name=config["TrainingConfig"].get("RunName",None),
                                        output_lable_dict_train=label_dict_train_path,
                                        output_lable_dict_test=label_dict_test_path)
            
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

        



        
        

        



                    
        