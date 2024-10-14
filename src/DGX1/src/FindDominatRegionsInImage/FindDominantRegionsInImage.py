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
    for image_path in image_paths:
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

    output_folder = "/".join(image_paths[0].split("/")[:len(image_paths[0].split("/")) - 1])

    mean_image -= 0.5
    mean_image = np.abs(mean_image)
 
    mean_image = mean_image * 255

    pil_image = Image.fromarray(np.uint8(mean_image))
    pil_image.save(f"{output_folder}/mean_image.jpg")


def findDominantRegionsInImagesbatched(dict_with_folders, num_threads=8):
    folders = list(dict_with_folders.keys())

    with open("./Errors.txt", "w") as f:
        f.write("")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(findDominantRegionsInImages, dict_with_folders[folder]) for folder in tqdm(folders, desc="Creating Threads")]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Waiting for threads",):
            try:
                future.result()
            except Exception as e:
                with open("./Errors.txt", "a") as f:
                    f.write(f"Error in put_locations_to_json_files: {e}\n")

        print("Done")
    
