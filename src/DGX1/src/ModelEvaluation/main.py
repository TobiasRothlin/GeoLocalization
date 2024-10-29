import torch
import os
import sys

RUN_PATH = "/home/tobias.rothlin/data/TrainingSnapshots/run_7"
MODEL_WEIGHTS =RUN_PATH + "/model_end_of_epoch_4.pt"

sys.path.append(RUN_PATH)
from Model import ClipLocationDecoder

sys.path.append("/home/tobias.rothlin/GeoLocalization/src/DGX1/src/ClipLocationDecoder")
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')
sys.path.append('/Users/tobiasrothlin/Documents/MSE/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from tqdm import tqdm
from random import randint

import traceback

import json

import dotenv
import mlflow

from GeoLocalizationDatasetDecoder import GeoLocalizationDatasetDecoder
from torch.utils.data import DataLoader

from HaversineLoss import HaversineLoss

from Evaluation import Evaluation


from torchinfo import summary

import matplotlib.pyplot as plt

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"
# BASE_PATH = "/Users/tobiasrothlin/Documents/MSE/Dataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")

TEST_DATA_FOLDER_IM2GPS = os.path.join(TEST_DATA_FOLDER, "img2gpsTestSet")
TEST_DATA_FOLDER_IM2GPS3K = os.path.join(TEST_DATA_FOLDER, "im2gps3ktest")
TEST_DATA_FOLDER_HOLDOUT = os.path.join(TEST_DATA_FOLDER, "GeoDataset2024")

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

    with open(RUN_PATH+"/config.json", "r") as f:
        configs = json.load(f)

    for config in configs["Runs"]:

        CHECK_IMAGE_FILES = False

        test_dataset_im2gps = GeoLocalizationDatasetDecoder(TEST_DATA_FOLDER_IM2GPS,
                                                            error_output="./error_output.txt",
                                                            standardization_coordinates=config["ModelConfig"]["StandardizationCoordinates"],
                                                            use_cache=True,
                                                            encoder_model=config["ModelConfig"]["BaseModel"])
        
        test_dataset_im2gps3k = GeoLocalizationDatasetDecoder(TEST_DATA_FOLDER_IM2GPS3K,
                                                            error_output="./error_output.txt",
                                                            standardization_coordinates=config["ModelConfig"]["StandardizationCoordinates"],
                                                            use_cache=True,
                                                            encoder_model=config["ModelConfig"]["BaseModel"])
        
        train_dataset_holdout = GeoLocalizationDatasetDecoder(TEST_DATA_FOLDER_HOLDOUT,
                                                            error_output="./error_output.txt",
                                                            standardization_coordinates=config["ModelConfig"]["StandardizationCoordinates"],
                                                            use_cache=True,
                                                            encoder_model=config["ModelConfig"]["BaseModel"])
        
        model = ClipLocationDecoder(standardization_coordinates=config["ModelConfig"]["StandardizationCoordinates"])

        model.load_from_checkpoint(MODEL_WEIGHTS)
        
        vector, label = train_dataset_holdout[0]
        summary(model, input_size=(1,vector.shape[0],vector.shape[1]))

        test_loader_im2gps = DataLoader(test_dataset_im2gps, 
                                        batch_size=config["TrainingConfig"]["TestBatchSize"], 
                                        shuffle=True,
                                        num_workers=config["TrainingConfig"]["NumWorkers"],
                                        persistent_workers=config["TrainingConfig"]["PersistantWorkers"], 
                                        prefetch_factor=config["TrainingConfig"]["PrefetchFactor"])
        
        test_loader_im2gps3k = DataLoader(test_dataset_im2gps3k, 
                                        batch_size=config["TrainingConfig"]["TestBatchSize"], 
                                        shuffle=True,
                                        num_workers=config["TrainingConfig"]["NumWorkers"],
                                        persistent_workers=config["TrainingConfig"]["PersistantWorkers"], 
                                        prefetch_factor=config["TrainingConfig"]["PrefetchFactor"])
        
        train_loader_holdout = DataLoader(train_dataset_holdout, 
                                        batch_size=config["TrainingConfig"]["TestBatchSize"], 
                                        shuffle=True,
                                        num_workers=config["TrainingConfig"]["NumWorkers"],
                                        persistent_workers=config["TrainingConfig"]["PersistantWorkers"], 
                                        prefetch_factor=config["TrainingConfig"]["PrefetchFactor"])

        loss_function = HaversineLoss(use_standarized_input=config["ModelConfig"]["StandardizationCoordinates"])

        evaluation_test_im2gps = Evaluation(model, test_loader_im2gps, device, loss_function)
        
        evaluation_test_im2gps3k = Evaluation(model, test_loader_im2gps3k, device, loss_function)
        
        evaluation_test_holdout = Evaluation(model, train_loader_holdout, device, loss_function)
        

        print("Evaluation Im2GPS")
        evaluation_test_im2gps.evaluate()
        evaluation_test_im2gps.to_file(RUN_PATH+"/evaluation_im2gps.txt")
        print(evaluation_test_im2gps)
        print(100*"=")
        print("Evaluation Im2GPS3K")
        evaluation_test_im2gps3k.evaluate()
        evaluation_test_im2gps3k.to_file(RUN_PATH+"/evaluation_im2gps3k.txt")
        print(evaluation_test_im2gps3k)
        print(100*"=")
        print("Evaluation Holdout")
        evaluation_test_holdout.evaluate()
        evaluation_test_holdout.to_file(RUN_PATH+"/evaluation_holdout.txt")
        print(evaluation_test_holdout)
        print(100*"=")

        try:
            dotenv.load_dotenv(dotenv.find_dotenv())

            os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

            mlflow.set_tracking_uri("https://mlflow.infs.ch")
            mlflow.set_experiment("GeoLocalization_Regression_Model")

            mlflow.start_run()
        
            mlflow.log_param("Model", config["ModelConfig"]["BaseModel"])
            mlflow.log_param("StandardizationCoordinates", config["ModelConfig"]["StandardizationCoordinates"])
            mlflow.log_param("TestBatchSize", config["TrainingConfig"]["TestBatchSize"])
            mlflow.log_param("NumWorkers", config["TrainingConfig"]["NumWorkers"])

            mlflow.log_param("TrainBatchSize", config["TrainingConfig"]["TrainBatchSize"])
            mlflow.log_param("Epochs", config["TrainingConfig"]["Epochs"])
            mlflow.log_param("LearningRate", config["TrainingConfig"]["LearningRate"])
            mlflow.log_param("WeightDecay", config["TrainingConfig"]["WeightDecay"])
            mlflow.log_param("PersistantWorkers", config["TrainingConfig"]["PersistantWorkers"])
            mlflow.log_param("PrefetchFactor", config["TrainingConfig"]["PrefetchFactor"])
            mlflow.log_param("GradientAccumulationSteps", config["TrainingConfig"]["GradientAccumulationSteps"])
            mlflow.log_param("Amsgrad", config["TrainingConfig"]["Amsgrad"])
            mlflow.log_param("Betas", config["TrainingConfig"]["Betas"])
            mlflow.log_param("Gamma", config["TrainingConfig"]["Gamma"])

            mlflow.log_metric("Im2GPS_Average_Loss", evaluation_test_im2gps.evaluation_results["average_loss"])
            mlflow.log_metric("Im2GPS3K_Average_Loss", evaluation_test_im2gps3k.evaluation_results["average_loss"])
            mlflow.log_metric("Holdout_Average_Loss", evaluation_test_holdout.evaluation_results["average_loss"])

            mlflow.log_metric("Im2GPS_Inside_1", evaluation_test_im2gps.evaluation_results["is_inside_average"][1])
            mlflow.log_metric("Im2GPS_Inside_25", evaluation_test_im2gps.evaluation_results["is_inside_average"][25])
            mlflow.log_metric("Im2GPS_Inside_200", evaluation_test_im2gps.evaluation_results["is_inside_average"][200])
            mlflow.log_metric("Im2GPS_Inside_750", evaluation_test_im2gps.evaluation_results["is_inside_average"][750])
            mlflow.log_metric("Im2GPS_Inside_2500", evaluation_test_im2gps.evaluation_results["is_inside_average"][2500])

            mlflow.log_metric("Im2GPS3K_Inside_1", evaluation_test_im2gps3k.evaluation_results["is_inside_average"][1])
            mlflow.log_metric("Im2GPS3K_Inside_25", evaluation_test_im2gps3k.evaluation_results["is_inside_average"][25])
            mlflow.log_metric("Im2GPS3K_Inside_200", evaluation_test_im2gps3k.evaluation_results["is_inside_average"][200])
            mlflow.log_metric("Im2GPS3K_Inside_750", evaluation_test_im2gps3k.evaluation_results["is_inside_average"][750])
            mlflow.log_metric("Im2GPS3K_Inside_2500", evaluation_test_im2gps3k.evaluation_results["is_inside_average"][2500])

            mlflow.log_metric("Holdout_Inside_1", evaluation_test_holdout.evaluation_results["is_inside_average"][1])
            mlflow.log_metric("Holdout_Inside_25", evaluation_test_holdout.evaluation_results["is_inside_average"][25])
            mlflow.log_metric("Holdout_Inside_200", evaluation_test_holdout.evaluation_results["is_inside_average"][200])
            mlflow.log_metric("Holdout_Inside_750", evaluation_test_holdout.evaluation_results["is_inside_average"][750])
            mlflow.log_metric("Holdout_Inside_2500", evaluation_test_holdout.evaluation_results["is_inside_average"][2500])

            mlflow.log_artifact(RUN_PATH+"/config.json")
            mlflow.log_artifact(RUN_PATH+"/Model.py")
            mlflow.log_artifact(MODEL_WEIGHTS)


        except Exception as e:
            print("Could not connect to MLFlow")
            print(e)
            traceback.print_exc()
            print("MLFlow disabled")

        mlflow.end_run()

        
        






                    
        