import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms as v2

from PIL import Image


from DataLocator import DataLocator

from tqdm import tqdm

import json
import os



class GeoLocalizationDataset(Dataset):
    def __init__(self, folder, image_mean, image_std,
                 image_width=None,
                 image_height=None,
                 augmentation=None,
                 use_center_crop=False,
                 error_output="./error_output.txt",
                 check_images=False):
        
        """
        :param folder: The folder containing the images and the labels
        :param image_width: The width of the image
        :param image_height: The height of the images
        :param augmentation: The augmentation pipeline
        :param use_center_crop: Whether to use center crop or resize
        :param error_output: The file to write the errors to
        :param check_images: Whether to check if the images exist and are valid
        """
        
        self.folder = folder
        self.check_images = check_images

        self.image_mean = image_mean
        self.image_std = image_std

        with open(error_output, "w") as f:
            f.write("")

        self.error_output = error_output

        self.image_width = image_width
        self.image_height = image_height

        if use_center_crop:
            self.resize_func = v2.CenterCrop(size=(image_height, image_width))
            print(f"Using center crop: {image_height}x{image_width}")
        else:
            self.resize_func = v2.Resize(size=(image_height, image_width))
            print(f"Using resize: {image_height}x{image_width}")
            print(self.resize_func)

        self.augmentation_pipeline = augmentation

        self.locator = DataLocator(folder)
        self.label_paths = self.locator.get_files(".json")
        
        self.data = []

        for json_file in tqdm(self.label_paths, desc="Loading data"):
            image, label = self.__load_data(json_file)
            if image is not None and label is not None:
                self.data.append((image, label))


    def __load_data(self,json_file):
        """
        Load the data from the json file
        :param json_file: The json file
        :return: The image path and the label
        """
        image_path = json_file.replace(".json", ".jpg")
        if not os.path.exists(image_path):
            image_path = json_file.replace(".json", ".jpeg")
        if not os.path.exists(image_path):
            with open(self.error_output, "a") as f:
                f.write(f"Image not found: {image_path}\n")
            return None, None
        
        if self.check_images:
            try:
                with Image.open(image_path) as img:
                    pass
            except Exception as e:
                with open(self.error_output, "a") as f:
                    f.write(f"Error opening image: {image_path}\n")
                return None, None

        with open(json_file, "r") as f:
            raw_data = json.load(f)

        return image_path, (raw_data["lat"], raw_data["lon"])
    
    def __augmentation(self, image):
        """
        Apply the augmentation pipeline
        :param image: The image
        :return: The augmented image
        """
        if self.augmentation_pipeline:
            image = self.augmentation_pipeline(image)
        return image
    
    def __load_image(self, image_path):
        """
        Load the image
        :param image_path: The image path
        :return: The image
        """
        with Image.open(image_path) as img:
            image = img.copy()
        image = image.convert("RGB")
        return image

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        
        image = self.__load_image(image_path)

        # Do some image processing
        if self.image_width and self.image_height:
            image = self.resize_func(image)

        if self.augmentation_pipeline:
            image = self.__augmentation(image)

        if isinstance(image, Image.Image):
            image = ToTensor()(image)
        elif isinstance(image, torch.Tensor):
            pass
        else:
            raise ValueError("Image type not supported")
        
        # Normalize the image
        image = (image - torch.tensor(self.image_mean).view(3, 1, 1)) / torch.tensor(self.image_std).view(3, 1, 1)

        label = torch.tensor(label)
        return image, label
    
    def get_raw_image(self, idx):
        """
        Get the raw image
        :param idx: The index
        :return: The image
        """
        image_path, _ = self.data[idx]
        image = self.__load_image(image_path)
        return image
    
    def get_image_location(self, idx):
        """
        Get the image location
        :param idx: The index
        :return: The image location
        """
        image_path, _ = self.data[idx]

        image, _ = self.__getitem__(idx)

        if "jpeg" in image_path:
            json_path = image_path.replace(".jpeg", ".json")
        elif "jpg" in image_path:
            json_path = image_path.replace(".jpg", ".json")
        else:
            raise ValueError("Image path not supported")

        with open(json_path, "r") as f:
            try:
                raw_data = json.load(f)
            except Exception as e:
                print(f"Error loading json file: {json_path}")
                raise e


        
        return image, raw_data["country"]
    
    