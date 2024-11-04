import torch
import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')
sys.path.append('/Users/tobiasrothlin/Documents/MSE/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from tqdm import tqdm
from random import randint

from CalculateImageEmbedding import ImageEmbeddingCalculator
from DataLocator import DataLocator

from concurrent.futures import ThreadPoolExecutor, as_completed

import json

import matplotlib.pyplot as plt

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"
# BASE_PATH = "/Users/tobiasrothlin/Documents/MSE/Dataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")

OUTPUT_FOLDER = "/home/tobias.rothlin/data/GeoDataset/ImageEmbeddings"


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

def get_cuda_divices():
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

def create_indipendent_batches(full_list, number_of_batches):
    batch_size = len(full_list) // number_of_batches
    remainder = len(full_list) % number_of_batches

    independent_batches = []
    start = 0

    for i in range(number_of_batches):
        end = start + batch_size + (1 if i < remainder else 0)
        independent_batches.append(full_list[start:end])
        start = end

    return independent_batches


def execute_image_embedding_calculation(data_set, cuda_device,config,id,output_path):
    print(f"Starting calculation on device {cuda_device} with data set of size {len(data_set)}")
    calculator = ImageEmbeddingCalculator(device=cuda_device,
                                          config=config,
                                          output_folder=output_path,
                                          json_files=data_set,
                                          thread_id=id)
    
    calculator.calculate_image_embedding(thread_id=id)
    return True

def run_image_embedding(dataset,config,output_folder):
    locator = DataLocator(dataset,use_cache=False)

    json_files = locator.get_files(".json")

    json_files = [f for f in json_files if "cache.json" not in f]

    model_config = config["ModelConfig"]

    cuda_devices = get_cuda_divices()

    independen_batches = create_indipendent_batches(json_files,len(cuda_devices))


    print(f"Starting {len(independen_batches)} independent batches")
    with ThreadPoolExecutor(max_workers=len(cuda_devices)) as executor:
        futures = {executor.submit(execute_image_embedding_calculation, batch, cuda_devices[i],model_config,i,output_folder): (batch,cuda_devices[i]) for i,batch in enumerate(independen_batches)}

        for future in as_completed(futures):
            print(f"Finished calculation: {future.result()}")


if __name__ == '__main__':
    
    device = checkCuda()

    with open("./config.json", "r") as f:
        configs = json.load(f)

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder {OUTPUT_FOLDER}")

    if not os.path.exists(OUTPUT_FOLDER+"/Test"):
        os.makedirs(OUTPUT_FOLDER+"/Test")
    
    if not os.path.exists(OUTPUT_FOLDER+"/Train"):
        os.makedirs(OUTPUT_FOLDER+"/Train")

    for config in configs["Runs"]:
        #run_image_embedding(TEST_DATA_FOLDER,config,OUTPUT_FOLDER+"/Test")
        run_image_embedding(TRAIN_DATA_FOLDER,config,OUTPUT_FOLDER+"/Train")