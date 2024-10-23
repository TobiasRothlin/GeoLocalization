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


def execute_image_embedding_calculation(data_set, cuda_device,config,id):
    print(f"Starting calculation on device {cuda_device} with data set of size {len(data_set)}")
    image_embedding_calculator = ImageEmbeddingCalculator(device=cuda_device,config=config,output_folder=OUTPUT_FOLDER)
    number_of_processed_files = 0
    elements_pre_batch = 2000
    current_batch = 0

    for json_file,image_path in tqdm(data_set, desc=f"Calculating embeddings id: {id}"):    
        image_embedding_calculator.calculate_image_embedding(image_path,json_file,subfolder=f"Batch_{id}_{current_batch}")
        number_of_processed_files += 1
        if number_of_processed_files % elements_pre_batch == 0:
            current_batch += 1


    return True

def run_image_embedding(dataset,config):
    locator = DataLocator(dataset,use_cache=False)

    json_files = locator.get_files(".json")

    data_set = []

    for json_file in tqdm(json_files, desc="Loading data"):
        image_path = json_file.replace(".json", ".jpg")
        if not os.path.exists(image_path):
            image_path = json_file.replace(".json", ".jpeg")
        if not os.path.exists(image_path):
            raise Exception(f"Image not found: {image_path}, {json_file}")
        data_set.append((json_file,image_path))
    

    model_config = config["ModelConfig"]

    cuda_devices = get_cuda_divices()

    independen_batches = create_indipendent_batches(data_set,len(cuda_devices))


    print(f"Starting {len(independen_batches)} independent batches")
    with ThreadPoolExecutor(max_workers=len(cuda_devices)) as executor:
        futures = {executor.submit(execute_image_embedding_calculation, batch, cuda_devices[i],model_config,i): (batch,cuda_devices[i]) for i,batch in enumerate(independen_batches)}

        for future in as_completed(futures):
            print(f"Finished calculation: {future.result()}")


if __name__ == '__main__':
    
    device = checkCuda()

    with open("./config.json", "r") as f:
        configs = json.load(f)

    for config in configs["Runs"]:
        run_image_embedding(TEST_DATA_FOLDER,config)
        run_image_embedding(TRAIN_DATA_FOLDER,config)