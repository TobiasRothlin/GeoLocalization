import torch
import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')
sys.path.append('/Users/tobiasrothlin/Documents/MSE/GeoLocalization/src/DGX1/src/Utility')
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/RegressionTraining')

from time import sleep

from tqdm import tqdm
from random import randint

import traceback

import json

import dotenv
import mlflow

from GeoLocalizationDataset import GeoLocalizationDataset
from torch.utils.data import DataLoader
from Model import GeoLocalizationModel

from Evaluation import Evaluation


from torchsummary import summary

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

    with open("./config.json", "r") as f:
        configs = json.load(f)

    for config in configs["Runs"]:

        CHECK_IMAGE_FILES = False

        test_dataset_im2gps = GeoLocalizationDataset(TEST_DATA_FOLDER_IM2GPS,
                                            image_width=config["ModelConfig"]["ImageWidth"],
                                            image_height=config["ModelConfig"]["ImageHeight"],
                                            use_center_crop=config["ModelConfig"]["UseCenterCrop"],
                                            check_images=CHECK_IMAGE_FILES,
                                            image_mean=config["ModelConfig"]["ImageMean"],
                                            image_std=config["ModelConfig"]["ImageStd"],
                                            standardization_coordinates=config["ModelConfig"]["StandardizationCoordinates"],
                                            use_cache=False)
        
        test_dataset_im2gps3k = GeoLocalizationDataset(TEST_DATA_FOLDER_IM2GPS3K,
                                            image_width=config["ModelConfig"]["ImageWidth"],
                                            image_height=config["ModelConfig"]["ImageHeight"],
                                            use_center_crop=config["ModelConfig"]["UseCenterCrop"],
                                            check_images=CHECK_IMAGE_FILES,
                                            image_mean=config["ModelConfig"]["ImageMean"],
                                            image_std=config["ModelConfig"]["ImageStd"],
                                            standardization_coordinates=config["ModelConfig"]["StandardizationCoordinates"],
                                            use_cache=False)
        
        train_dataset_holdout = GeoLocalizationDataset(TEST_DATA_FOLDER_HOLDOUT,
                                            image_width=config["ModelConfig"]["ImageWidth"],
                                            image_height=config["ModelConfig"]["ImageHeight"],
                                            use_center_crop=config["ModelConfig"]["UseCenterCrop"],
                                            check_images=CHECK_IMAGE_FILES,
                                            image_mean=config["ModelConfig"]["ImageMean"],
                                            image_std=config["ModelConfig"]["ImageStd"],
                                            standardization_coordinates=config["ModelConfig"]["StandardizationCoordinates"],
                                            use_cache=False)
        
        model = GeoLocalizationModel(config["ModelConfig"]["BaseModel"])

        if config["ModelConfig"]["LoadFromCheckpoint"]:
            state_dict = torch.load(config["ModelConfig"]["LoadFromCheckpoint"])
            model.load_state_dict(state_dict)
            print("Model loaded from: ", config["ModelConfig"]["LoadFromCheckpoint"])

        print("Base Model")
        summary(model.vision_model, (3, config["ModelConfig"]["ImageHeight"], config["ModelConfig"]["ImageWidth"]))
        print(100*"=")
        print("Full Model")
        summary(model, (3, config["ModelConfig"]["ImageHeight"], config["ModelConfig"]["ImageWidth"]))

        test_loader_im2gps = DataLoader(test_dataset_im2gps, batch_size=config["TestConfig"]["TestBatchSize"], shuffle=True,num_workers=config["TestConfig"]["NumWorkers"])
        test_loader_im2gps3k = DataLoader(test_dataset_im2gps3k, batch_size=config["TestConfig"]["TestBatchSize"], shuffle=True,num_workers=config["TestConfig"]["NumWorkers"])
        train_loader_holdout = DataLoader(train_dataset_holdout, batch_size=config["TestConfig"]["TestBatchSize"], shuffle=True,num_workers=config["TestConfig"]["NumWorkers"])

        evaluation_test_im2gps = Evaluation(model, test_loader_im2gps, device,user_standardized_input=config["ModelConfig"]["StandardizationCoordinates"])
        evaluation_test_im2gps3k = Evaluation(model, test_loader_im2gps3k, device,user_standardized_input=config["ModelConfig"]["StandardizationCoordinates"])

        evaluation_test_holdout = Evaluation(model, train_loader_holdout, device,user_standardized_input=config["ModelConfig"]["StandardizationCoordinates"])

        print("Evaluation Im2GPS")
        evaluation_test_im2gps.evaluate()
        print(evaluation_test_im2gps)
        print(100*"=")
        print("Evaluation Im2GPS3K")
        evaluation_test_im2gps3k.evaluate()
        print(evaluation_test_im2gps3k)
        print(100*"=")
        print("Evaluation Train")
        evaluation_test_holdout.evaluate()
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
            mlflow.log_param("ImageWidth", config["ModelConfig"]["ImageWidth"])
            mlflow.log_param("ImageHeight", config["ModelConfig"]["ImageHeight"])
            mlflow.log_param("UseCenterCrop", config["ModelConfig"]["UseCenterCrop"])
            mlflow.log_param("ImageMean", config["ModelConfig"]["ImageMean"])
            mlflow.log_param("ImageStd", config["ModelConfig"]["ImageStd"])
            mlflow.log_param("StandardizationCoordinates", config["ModelConfig"]["StandardizationCoordinates"])
            mlflow.log_param("LoadFromCheckpoint", config["ModelConfig"]["LoadFromCheckpoint"])
            mlflow.log_param("TestBatchSize", config["TestConfig"]["TestBatchSize"])
            mlflow.log_param("NumWorkers", config["TestConfig"]["NumWorkers"])
            mlflow.log_param("StandardizationCoordinates", config["ModelConfig"]["StandardizationCoordinates"])

            mlflow.log_metric("Im2GPS_Average_Loss", evaluation_test_im2gps.evaluation_results["average_loss"])
            mlflow.log_metric("Im2GPS3K_Average_Loss", evaluation_test_im2gps3k.evaluation_results["average_loss"])

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

            mlflow.log_artifact("./config.json")
            mlflow.log_artifact("/home/tobias.rothlin/GeoLocalization/src/DGX1/src/RegressionTraining/Model.py")
            mlflow.log_artifact(config["ModelConfig"]["LoadFromCheckpoint"])


        except Exception as e:
            print("Could not connect to MLFlow")
            print(e)
            traceback.print_exc()
            print("MLFlow disabled")

        mlflow.end_run()

        
        






                    
        