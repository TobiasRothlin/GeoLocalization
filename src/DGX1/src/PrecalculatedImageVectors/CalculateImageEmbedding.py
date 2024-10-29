import torch
import json
from PIL import Image
from torchvision import transforms as v2
from torchvision.transforms import ToTensor
from transformers import CLIPProcessor, CLIPModel
from torchsummary import summary
import os

import numpy as np


class ImageEmbeddingCalculator:
    def __init__(self,device,config,output_folder):

        base_model = CLIPModel.from_pretrained(config["BaseModel"])

        self.output_folder = output_folder

        self.embedding_folder = os.path.join(output_folder, config["BaseModel"].replace("/", "_"))
        if not os.path.exists(self.embedding_folder):
            os.makedirs(self.embedding_folder)
        else:
            print(f"Folder {self.embedding_folder} already exists")

        self.vision_model = base_model.vision_model

        # summary(self.vision_model, (3, config["ImageHeight"], config["ImageWidth"]))

        self.device = device
        self.config = config

        image_height = config["ImageHeight"]
        image_width = config["ImageWidth"]

        self.image_mean = config["ImageMean"]
        self.image_std = config["ImageStd"]

        self.vision_model = self.vision_model.to(device)

        if config["UseCenterCrop"]:
            self.resize_func = v2.CenterCrop(size=(image_height, image_width))
            # print(f"Using center crop: {image_height}x{image_width}")
        else:
            self.resize_func = v2.Resize(size=(image_height, image_width))
            # print(f"Using resize: {image_height}x{image_width}")

    def calculate_image_embedding(self, image_path,json_path,subfolder=None):
        
        with open(json_path, "r") as f:
            try:
                raw_data = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}")
                print(e)
                raise e
                return None
            
        if "image_embedding" not in raw_data:
            raw_data["image_embedding"] = {}

        if self.config["BaseModel"] in raw_data["image_embedding"]:
            return None
            
        image = self.__load_image(image_path)

        self.vision_model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            embedding = self.vision_model(image).last_hidden_state
            embedding = embedding.cpu().numpy()

        if subfolder is not None:
            subfolder_path = os.path.join(self.embedding_folder, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            if image_path.endswith(".jpeg"):
                vector_path = os.path.join(subfolder_path, os.path.basename(image_path).replace(".jpeg", ".npy"))
            else:
                vector_path = os.path.join(subfolder_path, os.path.basename(image_path).replace(".jpg", ".npy"))

            if "npy" not in vector_path:
                raise ValueError(f"Vector path not correct: {vector_path}") 
        else:
            if image_path.endswith(".jpeg"):
                vector_path = os.path.join(self.embedding_folder, os.path.basename(image_path).replace(".jpeg", ".npy"))
            else:
                vector_path = os.path.join(self.embedding_folder, os.path.basename(image_path).replace(".jpg", ".npy"))

            if "npy" not in vector_path:
                raise ValueError(f"Vector path not correct: {vector_path}") 

        np.save(vector_path, embedding)
        raw_data["image_embedding"][self.config["BaseModel"]] = vector_path
        with open(json_path, "w") as f:
            json.dump(raw_data, f, indent=4)

    
    def __load_image(self, image_path):
        image = Image.open(image_path)
        image = self.resize_func(image)
        image = ToTensor()(image)
        image = (image - torch.tensor(self.image_mean).view(3, 1, 1)) / torch.tensor(self.image_std).view(3, 1, 1)
        image_batch = image.unsqueeze(0)
        return image_batch
    

