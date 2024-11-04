import torch
import json
from PIL import Image
from torchvision import transforms as v2
from torchvision.transforms import ToTensor
from transformers import CLIPProcessor, CLIPModel
from torchsummary import summary
from torch.utils.data import Dataset
import os
from tqdm import tqdm

import numpy as np

class ImageEmbeddingDataset(Dataset):
    def __init__(self, json_files, base_model,thread_id):
        self.data = []
        self.config = base_model
        self.preprocessor = CLIPProcessor.from_pretrained(base_model)

        for json_file in tqdm(json_files, desc=f"Loading data {thread_id}"):
            image_path = json_file.replace(".json", ".jpg")
            if not os.path.exists(image_path):
                image_path = json_file.replace(".json", ".jpeg")
            if not os.path.exists(image_path):
                raise Exception(f"Image not found: {image_path}, {json_file}")
            self.data.append((json_file, image_path))


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        json_path,image_path = self.data[idx]
            
        image = Image.open(image_path)
        input_values = self.preprocessor(images=image)
        pixel_values = np.array(input_values["pixel_values"][0])
        pixel_values = torch.tensor(pixel_values, dtype=torch.float32)
        return pixel_values, json_path



class ImageEmbeddingCalculator:
    def __init__(self,device,config,output_folder,json_files,thread_id):
        base_model = CLIPModel.from_pretrained(config["BaseModel"])

        self.output_folder = output_folder

        self.embedding_folder = os.path.join(output_folder, config["BaseModel"].replace("/", "_"))
        if not os.path.exists(self.embedding_folder):
            os.makedirs(self.embedding_folder)
        else:
            print(f"Folder {self.embedding_folder} already exists")

        self.vision_model = base_model.vision_model
        self.dataset = ImageEmbeddingDataset(json_files=json_files, base_model=config["BaseModel"],thread_id=thread_id)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=200, shuffle=False,num_workers=6)

        # summary(self.vision_model, (3, config["ImageHeight"], config["ImageWidth"]))

        self.device = device
        self.config = config
        self.base_model = config["BaseModel"]

        self.vision_model = self.vision_model.to(device)
        self.vision_model.eval()


    def calculate_image_embedding(self,thread_id,max_batch_size=2000):
        number_of_processed_files = 0
        batch_index = 0
        output_path = os.path.join(self.embedding_folder, f"Batch_{thread_id}_{batch_index}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for images, json_paths in tqdm(self.data_loader,desc=f"Thread {thread_id}"):

            images = images.to(self.device)

            with torch.no_grad():
                vision_model_output = self.vision_model(images)
            
            for i in range(len(json_paths)):
                last_hidden_state = vision_model_output.last_hidden_state[i].cpu().numpy()
                pooler_output = vision_model_output.pooler_output[i].cpu().numpy()

                json_path = json_paths[i]

                with open(json_path, "r") as f:
                    json_data = json.load(f)

                json_file_name = os.path.basename(json_path).replace(".json", "")
                # last_hidden_state_path = os.path.join(output_path, f"{json_file_name.replace('/', '_')}_last_hidden_state.npy")
                pooler_output_path = os.path.join(output_path, f"{json_file_name.replace('/', '_')}_pooler_output.npy")

                try:
                    # np.save(last_hidden_state_path, last_hidden_state)
                    np.save(pooler_output_path, pooler_output)
                except Exception as e:
                    print(f"Error in saving file: {e}")
                    # print(f"Path: {last_hidden_state_path}")
                    print(f"Path: {pooler_output_path}")
                    print(f"Json path: {json_path}")
                    print(f"Json data: {json_data}")
                    raise e

                json_data["image_embedding"] = {
                    self.base_model: {}
                }

                # json_data["image_embedding"][self.base_model]["last_hidden_state"] = last_hidden_state_path
                json_data["image_embedding"][self.base_model]["pooler_output"] = pooler_output_path
                
                with open(json_path, "w") as f:
                    json.dump(json_data, f, indent=4)

                number_of_processed_files += 1

                if number_of_processed_files % max_batch_size == 0:
                    batch_index += 1
                    output_path = os.path.join(self.embedding_folder, f"Batch_{thread_id}_{batch_index}")
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)

