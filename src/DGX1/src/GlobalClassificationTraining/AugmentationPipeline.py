from torchvision import transforms as v2
from PIL import Image

import matplotlib.pyplot as plt

EXAMPLE_IMAGE = "/home/tobias.rothlin/data/GeoDataset/Test/GeoDataset2024/_2_AO6mBLazi6WkAk70C-w.jpg"

def get_augmentation_pipeline(augmentaions,use_Example_Image=None):
    if augmentaions is None:
        return None
    
    pipeline = v2.Compose([do_mapping(aug["name"],aug["params"]) for aug in augmentaions])

    if not use_Example_Image is None:
        image = Image.open(EXAMPLE_IMAGE)

        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(pipeline(image))

        plt.savefig("./augmentation_preview.png")
        plt.close()

    return pipeline

        


def do_mapping(layer,params):
    if layer == "RandomRotation":
        return v2.RandomRotation(degrees=params["degrees"])
    
    elif layer == "ColorJitter":
        return v2.ColorJitter(brightness=params["brightness"], contrast=params["contrast"], saturation=params["saturation"], hue=params["hue"])

    elif layer == "RandomPerspective":
        return v2.RandomPerspective(distortion_scale=params["distortion_scale"], p=params["p"])
    else:
        raise ValueError(f"Layer {layer} not implemented")
    
        
   