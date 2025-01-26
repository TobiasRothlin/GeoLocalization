import torch
import os
import sys

EvalConfigPath = "/home/tobias.rothlin/GeoLocalization/src/DGX1/src/HybridModelEvaluation/config.json"
DataLoaderConfigPath = "/home/tobias.rothlin/GeoLocalization/src/DGX1/src/HybridModelEvaluation/datasetConfig.json"

sys.path.append("/home/tobias.rothlin/GeoLocalization/src/DGX1/src/EncoderFineTuning")
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')
sys.path.append('/Users/tobiasrothlin/Documents/MSE/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from tqdm import tqdm
from random import randint

import traceback

import json

import dotenv
import mlflow

from GeoLocalizationDataset import GeoLocalizationDataset

from torch.utils.data import DataLoader

from Model import LocationDecoder
from ClassificationModel import LocationDecoder as LocationDecoderClassification
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
TEST_DATA_FOLDER_GWS15K = os.path.join(TEST_DATA_FOLDER, "GWS15K")

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

    with open(DataLoaderConfigPath, "r") as f:
        dataLoader_config = json.load(f)

    with open(EvalConfigPath, "r") as f:
        run_config = json.load(f)

    with open(run_config["Model"]["Classification"]["ModelPath"]+run_config["Model"]["Classification"]["ModelConfig"], "r") as f:
        model_config = json.load(f)

    with open(run_config["Model"]["Classification"]["ModelPath"]+"/output_lable_dict_test.json", "r") as f:
        label_to_index = json.load(f)
    
    index_to_label = {v: k for k, v in label_to_index.items()}

    print("Index to Label")
    print(index_to_label)


    dataLoader_config["DatasetConfig"]["augmentaion_pipeline"] = None

    CHECK_IMAGE_FILES = False

    test_dataset_im2gps = GeoLocalizationDataset(TEST_DATA_FOLDER_IM2GPS,**dataLoader_config["DatasetConfig"])
    
    test_dataset_im2gps3k = GeoLocalizationDataset(TEST_DATA_FOLDER_IM2GPS3K,**dataLoader_config["DatasetConfig"])
    
    train_dataset_holdout = GeoLocalizationDataset(TEST_DATA_FOLDER_HOLDOUT,**dataLoader_config["DatasetConfig"])

    test_dataset_gws15k = GeoLocalizationDataset(TEST_DATA_FOLDER_GWS15K,**dataLoader_config["DatasetConfig"])

    classification_model = LocationDecoderClassification(model_config["ModelConfig"],
                                    base_model=model_config["DatasetConfig"]["base_model"],
                                    use_pre_calculated_embeddings=model_config["DatasetConfig"]["use_pre_calculated_embeddings"],
                                    freeze_base_model=model_config["ModelConfig"]["freeze_base_model"],)
    
    classification_model.summary()
    classification_model.load(run_config["Model"]["Classification"]["ModelPath"]+run_config["Model"]["Classification"]["ModelWeights"])
    
    vector, label = train_dataset_holdout[0]
    print(vector.shape)
    print(label)
    summary(classification_model, input_size=(1,*vector.shape))

    regression_models = {}
    for regression_model_location in run_config["Model"]["Regression"]:
        with open(run_config["Model"]["Regression"][regression_model_location]["ModelPath"]+run_config["Model"]["Regression"][regression_model_location]["ModelConfig"], "r") as f:
            run_model_config = json.load(f)

        regression_model = LocationDecoder(run_model_config["ModelConfig"],
                                    base_model=run_model_config["DatasetConfig"]["base_model"],
                                    use_pre_calculated_embeddings=run_model_config["DatasetConfig"]["use_pre_calculated_embeddings"],
                                    freeze_base_model=run_model_config["ModelConfig"]["freeze_base_model"],)
        print(f"--{regression_model_location}--")
        regression_model.summary()
        regression_model.load(run_config["Model"]["Regression"][regression_model_location]["ModelPath"]+run_config["Model"]["Regression"][regression_model_location]["ModelWeights"])
        regression_models[regression_model_location] = regression_model
        regression_model.to("cpu")

    test_loader_im2gps = DataLoader(test_dataset_im2gps, 
                                    **dataLoader_config["DataLoaderConfig"]["Test"])
    
    test_loader_im2gps3k = DataLoader(test_dataset_im2gps3k, 
                                    **dataLoader_config["DataLoaderConfig"]["Test"])
    
    train_loader_holdout = DataLoader(train_dataset_holdout, 
                                    **dataLoader_config["DataLoaderConfig"]["Test"])
    
    test_loader_gws15k = DataLoader(test_dataset_gws15k, **dataLoader_config["DataLoaderConfig"]["Test"])

    loss_function = HaversineLoss(use_standarized_input=True)

    evaluation_test_im2gps = Evaluation(classification_model,index_to_label, regression_models,test_loader_im2gps, device, loss_function)
    
    evaluation_test_im2gps3k = Evaluation(classification_model,index_to_label, regression_models, test_loader_im2gps3k, device, loss_function)
    
    evaluation_test_holdout = Evaluation(classification_model,index_to_label, regression_models, train_loader_holdout, device, loss_function)

    evaluation_test_gws15k = Evaluation(classification_model,index_to_label, regression_models, test_loader_gws15k, device, loss_function)
    

    print("Evaluation Im2GPS")
    evaluation_test_im2gps.evaluate()
    evaluation_test_im2gps.to_file(run_config["Model"]["Classification"]["ModelPath"]+"/evaluation_im2gps.txt")
    print(evaluation_test_im2gps)
    print(100*"=")
    print("Evaluation Im2GPS3K")
    evaluation_test_im2gps3k.evaluate()
    evaluation_test_im2gps3k.to_file(run_config["Model"]["Classification"]["ModelPath"]+"/evaluation_im2gps3k.txt")
    print(evaluation_test_im2gps3k)
    print(100*"=")
    print("Evaluation Holdout")
    evaluation_test_holdout.evaluate()
    evaluation_test_holdout.to_file(run_config["Model"]["Classification"]["ModelPath"]+"/evaluation_holdout.txt")
    print(evaluation_test_holdout)
    print(100*"=")
    print("Evaluation GWS15K")
    evaluation_test_gws15k.evaluate()
    evaluation_test_gws15k.to_file(run_config["Model"]["Classification"]["ModelPath"]+"/evaluation_gws15k.txt")
    print(evaluation_test_gws15k)
    print(100*"=")

    try:
        dotenv.load_dotenv(dotenv.find_dotenv())

        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

        mlflow.set_tracking_uri("https://mlflow.infs.ch")
        mlflow.set_experiment("GeoLocalization_Regression_Model")

        mlflow.start_run()

        mlflow.log_param("Run Name", run_config["Model"]["Classification"]["ModelPath"].split("/")[-1])

        mlflow.log_metric("Im2GPS_Average_Loss", evaluation_test_im2gps.evaluation_results["average_loss"])
        mlflow.log_metric("Im2GPS3K_Average_Loss", evaluation_test_im2gps3k.evaluation_results["average_loss"])
        mlflow.log_metric("Holdout_Average_Loss", evaluation_test_holdout.evaluation_results["average_loss"])
        mlflow.log_metric("GWS15K_Average_Loss", evaluation_test_gws15k.evaluation_results["average_loss"])

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

        mlflow.log_metric("GWS15K_Inside_1", evaluation_test_gws15k.evaluation_results["is_inside_average"][1])
        mlflow.log_metric("GWS15K_Inside_25", evaluation_test_gws15k.evaluation_results["is_inside_average"][25])
        mlflow.log_metric("GWS15K_Inside_200", evaluation_test_gws15k.evaluation_results["is_inside_average"][200])
        mlflow.log_metric("GWS15K_Inside_750", evaluation_test_gws15k.evaluation_results["is_inside_average"][750])
        mlflow.log_metric("GWS15K_Inside_2500", evaluation_test_gws15k.evaluation_results["is_inside_average"][2500])


        mlflow.log_artifact(run_config)
        # mlflow.log_artifact(MODEL_WEIGHTS)


    except Exception as e:
        print("Could not connect to MLFlow")
        print(e)
        traceback.print_exc()
        print("MLFlow disabled")

    mlflow.end_run()

        
        






                    
        