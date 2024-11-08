import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')
sys.path.append('/Users/tobiasrothlin/Documents/MSE/GeoLocalization/src/DGX1/src/Utility')

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms as v2
import numpy as np
from transformers import CLIPProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from time import sleep
import time

from DataLocator import DataLocator
from createdBatches import create_indipendent_batches
from AugmentationPipeline import get_augmentation_pipeline

from tqdm import tqdm

import json
import os

class GeoLocalizationDataset(Dataset):
    def __init__(self,
                    dataset_folder,
                    base_model,
                    augmentaion_pipeline=None,
                    normalize_labels=True,
                    use_cached_dataloader=False,
                    load_for_contrast_learning=False,
                    use_pre_calculated_embeddings=False,
                    load_pooling_output=False,
                    workers=16,
                    error_file="error_file_dataset.txt",
                    timing_log="timing_log.txt",
                    use_gaussian_smoothing=False):
                 
        self.dataset_folder = dataset_folder

        self.augmentaion_pipeline = get_augmentation_pipeline(augmentaion_pipeline,use_Example_Image=True)

        self.normalize_labels = normalize_labels

        self.base_model = base_model
        self.preprocessor = CLIPProcessor.from_pretrained(base_model)

        self.use_cached_dataloader = use_cached_dataloader

        self.load_for_contrast_learning = load_for_contrast_learning

        self.use_pre_calculated_embeddings = use_pre_calculated_embeddings
        self.load_pooling_output = load_pooling_output

        self.workers = workers

        self.use_gaussian_smoothing = use_gaussian_smoothing

        self.error_file = error_file

        with open(self.error_file, "w") as f:
            f.write("")

        if not os.path.exists(timing_log):
            with open(timing_log, "w") as f:
                f.write("")
                
        
        if not os.path.exists(self.dataset_folder):
            raise ValueError(f"Folder {self.dataset_folder} does not exist")
        
        if self.augmentaion_pipeline is not None:
            print("Using augmentation pipeline")
            print(self.augmentaion_pipeline)
        else:
            print("No augmentation pipeline")

        
        if self.normalize_labels:
            print("Normalizing labels")
        else:
            print("Not normalizing labels")

        print(f"Using {self.workers} workers to load data")
        self.load_data()

    def load_data(self):
        data_locator = DataLocator(self.dataset_folder, self.workers,self.use_cached_dataloader)

        json_files = data_locator.get_files(".json")

        json_files = [f for f in json_files if "cache.json" not in f]

        batched_json_files = create_indipendent_batches(json_files, self.workers)

        self.data = []
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self.__process_json_files, batch, idx) for idx, batch in enumerate(batched_json_files)]
            for future in as_completed(futures):
                self.data.extend(future.result())

        end_time = time.time()

        with open("timing_log.txt", "a") as f:
            f.write(f"Workers:{self.workers},{end_time - start_time}s\n")

        sleep(1)
        print("")
        print("")
        print(100*"=")
        print(f"Loaded {len(self.data)} data points")
        print(100*"=")
        

    def __len__(self):
            return len(self.data)
        
    def __getitem__(self, idx):
        if self.load_for_contrast_learning:
            idx_1 = torch.randint(0, len(self.data), (1,)).item()
            idx_2 = torch.randint(0, len(self.data), (1,)).item()
            image_path_1, label_1 = self.data[idx_1]
            image_path_2, label_2 = self.data[idx_2]

            if self.use_pre_calculated_embeddings:
                image_1 = self.__load_vector(image_path_1)
                image_2 = self.__load_vector(image_path_2)
            else:
                image_1 = self.__load_image(image_path_1)
                image_2 = self.__load_image(image_path_2)

            if self.use_gaussian_smoothing:
                label_1 = self.__gaussian_smoothing_lat_lon(label_1[0], label_1[1])
                label_2 = self.__gaussian_smoothing_lat_lon(label_2[0], label_2[1])
            

            if self.normalize_labels:
                label_1 = self.__normalize_labels(label_1)
                label_2 = self.__normalize_labels(label_2)

            
            label_1 = torch.tensor([label_1[0],label_1[1]], dtype=torch.float32)
            label_2 = torch.tensor([label_2[0],label_2[1]], dtype=torch.float32)

            return image_1, image_2, label_1, label_2


        else:
            image_path, label = self.data[idx]

            if self.use_pre_calculated_embeddings:
                image = self.__load_vector(image_path)
            else:
                image = self.__load_image(image_path)

            if self.use_gaussian_smoothing:
                label = self.__gaussian_smoothing_lat_lon(label[0], label[1])

            if self.normalize_labels:
                label = self.__normalize_labels(label)      

            label = torch.tensor([label[0],label[1]], dtype=torch.float32)
            return image, label
        
    def __normalize_labels(self, label):
        return (label[0] / 90, label[1] / 180)


        
    def __load_image(self, image_path):
        image = Image.open(image_path)

        if self.augmentaion_pipeline is not None:
            image = self.augmentaion_pipeline(image)
        
        model_input = self.preprocessor(images=image)

        pixel_values = torch.tensor(model_input['pixel_values'][0], dtype=torch.float32)

        # Remove the batch dimension
        return pixel_values
    
    def __load_vector(self, numpy_path):
        vector = np.load(numpy_path)
        vector = torch.tensor(vector, dtype=torch.float32)
        return vector

    def __process_json_files(self, json_files,batch_idx):
        collected_data = []

        progress_bar = tqdm(json_files,total=len(json_files), desc=f"Batch {batch_idx}")

        running_index = 0
        for json_file in progress_bar:
            try:
                image_path, label = self.__process_json_file(json_file)
                collected_data.append((image_path, label))

            except Exception as e:
                with open(self.error_file, "a") as f:
                    f.write(f"Error in {json_file}: {e}\n")
            running_index += 1
            progress_bar.postfix = f"Collected: {(len(collected_data)/running_index)*100:.2f}%"
            progress_bar.update(1)

        progress_bar.close()

        return collected_data
    


    
    def __process_json_file(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        
        lat = data["lat"]
        lon = data["lon"]

        if self.use_pre_calculated_embeddings:
            if self.load_pooling_output:
                image_embedding_path = data["image_embedding"][self.base_model]["pooler_output"]
            else:
                image_embedding_path = data["image_embedding"][self.base_model]["last_hidden_state"]
            
            if os.path.exists(image_embedding_path):
                return image_embedding_path, (lat, lon)
            else:
                raise ValueError(f"Embedding path not found: {json_file}")
            
        else:
            image_path = self.__get_valid_image_path(json_file)
            if os.path.exists(image_path):
                return image_path, (lat, lon)
            else:
                raise ValueError(f"Image path not found: {json_file}")

    def __get_valid_image_path(self,json_path):
        image_path = json_path.replace(".json", ".jpg")

        if os.path.exists(image_path):
            return image_path
        
        image_path = json_path.replace(".json", ".jpeg")

        if os.path.exists(image_path):
            return image_path
        
        raise ValueError(f"Matching Image path not found: {json_path}")
    
    def __gaussian_smoothing_lat_lon(self, lat, lon, scale = 1000):
        lat = lat + np.random.normal(0, 1)/scale
        lon = lon + np.random.normal(0, 1)/scale
        return lat, lon
        
        
        
        