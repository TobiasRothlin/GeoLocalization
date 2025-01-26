import numpy as np
from PIL import Image

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

def calculate_mean_of_images(image_paths):
    """
    Calculate the mean of the images
    :param image_paths: The paths to the images
    :return: The mean of the images
    """

    mean = np.zeros((336, 336, 3))
    for image_path in tqdm(image_paths, desc="Calculating Mean"):
        image = Image.open(image_path)

        image = image.resize((336, 336))


        image = np.array(image)
        image = image / 255.0
        mean += image/len(image_paths)

    # Convert to gray scale
    mean = np.mean(mean, axis=2)
    return mean


def findDominantRegionsInImages(image_paths):
    mean_image = calculate_mean_of_images(image_paths)

 
    mean_image = mean_image * 255


    pil_image = Image.fromarray(np.uint8(mean_image))
    return pil_image

    
