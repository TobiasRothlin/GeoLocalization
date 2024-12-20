import torch
from torch.utils.data import Dataset
from DataLocator import DataLocator

from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

import numpy as np

import json
import os

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



class GeoLocalizationDatasetDecoder(Dataset):

    def __init__(self, folder,
                 error_output="./error_output.txt",
                 standardization_coordinates=False,
                 use_cache=True,
                 encoder_model=None):
        
        """
        :param folder: The folder containing the images and the labels
        :param error_output: The file to write the errors to
        :param standardization_coordinates: Whether to standardize the coordinates
        :param use_cache: Whether to use the cache
        :param encoder_model: The encoder model to use
        :param std_dev_km: The standard deviation for the gaussian smoothing

        """
        self.EARTH_RADIUS = 6371.0
        self.PI = torch.acos(torch.zeros(1)).item() * 2
        self.folder = folder

        self.encoder_model = encoder_model

        self.standardization_coordinates = standardization_coordinates

        with open(error_output, "w") as f:
            f.write("")

        self.error_output = error_output

        self.data_locator = DataLocator(folder, use_cache=use_cache)

        self.json_files = self.data_locator.get_files(".json")

        self.data = []

        self.data = self.__load_data_threaded(self.json_files)

        print(f"Loaded {len(self.data)} files")

    def __load_vector(self, numpy_path):
        vector = np.load(numpy_path)
        # Remove batch dimension
        vector = vector[0, :]
        return vector
    
    def __clean_numpy_path(self, numpy_path):
        if numpy_path.endswith(".jpg"):
            numpy_path = numpy_path.replace(".jpg", ".npy")
        elif numpy_path.endswith(".jpeg"):
            numpy_path = numpy_path.replace(".jpeg", ".npy")
        return numpy_path
    
    def __load_data_threaded(self,json_file,workers=24):
        independent_batches = create_indipendent_batches(json_file,workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for idx,batch in enumerate(independent_batches):
                future = executor.submit(self.__load_data_batch, batch,idx)
                futures.append(future)
            results = []
            for future in as_completed(futures):
                results.extend(future.result())

        return results
        
    def __load_data_batch(self,json_files,idx=None):
        data_list = []
        for json_file in tqdm(json_files, desc=f"Loading data {idx}"):
            data = self.__load_file(json_file)
            if data is not None:
                data_list.append(data)
        return data_list


    def __load_file(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        try:
            lat = data["lat"]
            lon = data["lon"]

            vector_path = data["image_embedding"][self.encoder_model]

            vector_path = self.__clean_numpy_path(vector_path)

            if not os.path.exists(vector_path):
                with open(self.error_output, "a") as f:
                    f.write(f"Vector not found: {vector_path} in file: {json_file}\n")
                return None

            return vector_path, torch.tensor([lat, lon])
        except Exception as e:
            with open(self.error_output, "a") as f:
                f.write(f"Error loading file: {json_file}\n")
                f.write(str(e))
                f.write("\n")
            return None



    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vector_path , label = self.data[idx]
        vector = self.__load_vector(vector_path)
        if self.standardization_coordinates:
            label = self.__standardize_coordinates(label)
        return vector, label
    
    
    def __standardize_coordinates(self,label):
        return torch.tensor([label[0].item() / 90, label[1].item() / 180])

