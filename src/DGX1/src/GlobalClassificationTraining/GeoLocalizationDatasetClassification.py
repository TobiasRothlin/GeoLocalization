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

class GeoLocalizationClassificationDataset(Dataset):
    def __init__(self,
                    dataset_folder,
                    base_model,
                    augmentaion_pipeline=None,
                    normalize_labels=True,
                    use_cached_dataloader=False,
                    use_pre_calculated_embeddings=False,
                    load_pooling_output=False,
                    workers=16,
                    error_file="error_file_dataset.txt",
                    timing_log="timing_log.txt",
                    limit_data_by_continent=None,
                    limit_data_by_country=None,
                    do_classification_by_continent=False,
                    do_classification_by_country=False):
                 
        self.dataset_folder = dataset_folder

        if do_classification_by_continent and do_classification_by_country:
            raise ValueError("Cannot classify by continent and country at the same time")
        elif not do_classification_by_continent and not do_classification_by_country:
            raise ValueError("Must classify by continent or country")
        

        self.augmentaion_pipeline = get_augmentation_pipeline(augmentaion_pipeline,use_Example_Image=True)

        self.normalize_labels = normalize_labels

        self.base_model = base_model
        self.preprocessor = CLIPProcessor.from_pretrained(base_model)

        self.use_cached_dataloader = use_cached_dataloader

        self.use_pre_calculated_embeddings = use_pre_calculated_embeddings
        self.load_pooling_output = load_pooling_output

        self.workers = workers

        self.error_file = error_file

        self.limit_data_by_continent = limit_data_by_continent
        self.limit_data_by_country = limit_data_by_country

        self.do_classification_by_continent = do_classification_by_continent
        self.do_classification_by_country = do_classification_by_country

        if self.limit_data_by_continent is not None:
            print(f"Limiting data to continent: {self.limit_data_by_continent}")

        if self.limit_data_by_country is not None:
            print(f"Limiting data to country: {self.limit_data_by_country}")

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

        self.label_dict = {}
        self.label_histogram = {}

        for _, label in self.data:
            if label not in self.label_dict:
                self.label_dict[label] = len(self.label_dict)
                self.label_histogram[label] = 1
            else:
                self.label_histogram[label] += 1

        with open("timing_log.txt", "a") as f:
            f.write(f"Workers:{self.workers},{end_time - start_time}s\n")

        sleep(1)
        print("")
        print("")
        print(100*"=")
        print(f"Loaded {len(self.data)} data points")
        print(f"Unique labels: {len(self.label_dict)}")
        print(f"Labels dict: {self.label_dict}")
        print(f"Label histogram: {self.label_histogram}")
        print(100*"=")
        
    def get_label_dict(self):
        return self.label_dict
    
    def set_label_dict(self, label_dict):
        for key in self.label_dict:
            if key not in label_dict:
                raise ValueError(f"Key {key} not found in new label dict")
        self.label_dict = label_dict

    def save_label_dict(self, path):
        with open(path, "w") as f:
            json.dump(self.label_dict, f, indent=4)
    
    def get_idx_dict(self):
        return {v:k for k,v in self.label_dict.items()}
    
    def get_label_histogram(self):
        hist_vector = [self.label_histogram[self.get_idx_dict()[idx]] for idx in range(len(self.label_dict))]
        return torch.tensor(hist_vector, dtype=torch.float32)
    
    def __len__(self):
            return len(self.data)
        
    def __getitem__(self, idx):
        image_path, label = self.data[idx]

        if self.use_pre_calculated_embeddings:
            image = self.__load_vector(image_path)
        else:
            image = self.__load_image(image_path)     

        label = self.__one_hot_encode(label)
        return image, label
    
    def __one_hot_encode(self, label):
        vec = np.zeros(len(self.label_dict))
        vec[self.label_dict[label]] = 1
        return torch.tensor(vec, dtype=torch.float32)

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
                if image_path is not None and label is not None:
                    collected_data.append((image_path, label))

            except Exception as e:
                with open(self.error_file, "a") as f:
                    f.write(f"Error in {json_file}: {e}\n")
            running_index += 1
            progress_bar.postfix = f"Collected: {(len(collected_data)/running_index)*100:.2f}%"
            progress_bar.update(1)

        progress_bar.close()

        if self.limit_data_by_continent is not None:
            print(f"Filtered {len(json_files) - len(collected_data)} by continent")

        if self.limit_data_by_country is not None:
            print(f"Filtered {len(json_files) - len(collected_data)} by country")

        return collected_data
    
    def __process_json_file(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        
        if self.do_classification_by_continent:
            location = data["continent"]
        elif self.do_classification_by_country:
            location = data["country"]
        else:
            raise ValueError("Unknown classification type")

        if self.limit_data_by_continent is not None:
            if data["continent"] != self.limit_data_by_continent:
                return None, None
        
        if self.limit_data_by_country is not None:
            if data["country"] != self.limit_data_by_country:
                return None, None

        if self.use_pre_calculated_embeddings:
            if self.load_pooling_output:
                image_embedding_path = data["image_embedding"][self.base_model]["pooler_output"]
            else:
                image_embedding_path = data["image_embedding"][self.base_model]["last_hidden_state"]
            
            if os.path.exists(image_embedding_path):
                return image_embedding_path, location
            else:
                raise ValueError(f"Embedding path not found: {json_file}")
        else:
            image_path = self.__get_valid_image_path(json_file)
            if os.path.exists(image_path):
                return image_path, location
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
    
        
        
        