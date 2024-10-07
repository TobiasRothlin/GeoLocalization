from PIL import Image

from transformers import CLIPProcessor, CLIPModel

from tqdm import tqdm

from PIL import Image

import json

import os

class Classifier:
    def __init__(self,device="cpu"):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()
        self.device = device
        self.model = self.model.to(self.device)


    def classify(self, image_path,classes):
        image = Image.open(image_path)
        inputs = self.processor(text=classes, images=image, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

        argmax = probs.argmax()

        return classes[argmax]

def runClassification(json_paths, classes, device):
    classifier = Classifier(device=device)

    for json_path in tqdm(json_paths, desc="Classifying Images"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            if "DidClassification" in data and data["DidClassification"]:
                continue

            image_path = json_path.replace(".json", ".jpg")

            if not os.path.exists(image_path):
                image_path = json_path.replace(".json", ".jpeg")
                if not os.path.exists(image_path):
                    print(f"Image {image_path} does not exist")
                    continue


            try:
                image = Image.open(image_path)
            except:
                with open("classification_errors_could_not_open.txt", "a") as f:
                    f.write(f"{json_path}\n")
                continue

            predictedClass_regions = classifier.classify(image_path, classes["Regions"])
            predictedClass_populationAreas = classifier.classify(image_path, classes["PopulationAreas"])

            data["PredictedRegion"] = predictedClass_regions
            data["PredictedPopulationArea"] = predictedClass_populationAreas
            data["DidClassification"] = True

            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            with open("classification_errors.txt", "a") as f:
                f.write(f"{json_path}: {str(e)}\n")

        