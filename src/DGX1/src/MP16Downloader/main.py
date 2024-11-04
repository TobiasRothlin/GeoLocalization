import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from tqdm import tqdm

import json

import pandas as pd

import requests

from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")

DOWNLOAD_CSV_PATH = "/home/tobias.rothlin/data/GeoDataset/MP16_Pro_filtered_with_url.csv"

ERROR_FILE = "./error.txt"

def remove_cached_files():
    if os.path.exists("/home/tobias.rothlin/data/GeoDataset/Test/cache.json"):
        os.remove("/home/tobias.rothlin/data/GeoDataset/Test/cache.json")
        print("Removed cache file for Test data")
    else:
        print("No cache file found")

    if os.path.exists("/home/tobias.rothlin/data/GeoDataset/Train/cache.json"):
        os.remove("/home/tobias.rothlin/data/GeoDataset/Train/cache.json")
        print("Removed cache file for Train data")
    else:
        print("No cache file found")


def get_unque_folder_name(folder_name_og, base_folder):
    i = 0
    folder_name = f"{folder_name_og}_{i}"
    while os.path.exists(os.path.join(base_folder, folder_name)):
        folder_name = f"{folder_name_og}_{i}"
        i+=1

    new_path = os.path.join(base_folder, folder_name)
    
    try:
        os.makedirs(new_path)
    except Exception as e:
        print(f"Error in creating folder {new_path}: {e}")
        new_path = get_unque_folder_name(folder_name_og, base_folder)
    return new_path

def load_image(url,idx=None):
    if idx is not None:
        idx += 1
    response = requests.get(url)
    if response.status_code != 200:
        sleep(idx*5)
        load_image(url,idx)
    return response.content

def load_single_image(lat,lon, folder_path, image_name, url):
    
    try:
        image_path = os.path.join(folder_path, f"{image_name}.jpg")
        image = load_image(url)
        with open(image_path, "wb") as f:
            f.write(image)
    except Exception as e:
        with open(ERROR_FILE, "a") as f:
            f.write(f"Error in {image_name},{url}: {e}\n")
        return

    json_data = {
        "lat": lat,
        "lon": lon,
        "url": url,
    }
    json_path = os.path.join(folder_path, f"{image_name}.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f)

def process_batch(batch, batch_idx):
    sleep(0.1)
    folder_path = get_unque_folder_name(f"MP16", TRAIN_DATA_FOLDER)
    image_index = 0
    for element in tqdm(batch, total=len(batch), desc=f"DownloadingBatch {batch_idx}"):
        lat = element["lat"]
        lon = element["lon"]
        url = element["url"]
        image_name = f"MP16_{batch_idx}_{image_index}"
        load_single_image(lat, lon, folder_path, image_name, url)
        image_index+=1

def download_images(batched_elements,workers=16):
    print(f"Downloading {len(batched_elements):,d} images")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_batch, batch, idx): batch for idx, batch in enumerate(batched_elements)}
        for future in as_completed(futures):
            future.result()
    

if __name__ == "__main__":    
    remove_cached_files()

    with open(ERROR_FILE, "w") as f:
        f.write("")

    BATCH_SIZE = 2000
    df = pd.read_csv(DOWNLOAD_CSV_PATH)


    elements = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preprocessing images"):
        try:
            lat = row["LAT"]
            lon = row["LON"]
            url = row["download_url"]
            elements.append({
                "lat": lat,
                "lon": lon,
                "url": url,
            })

        except Exception as e:
            print(f"Error in row {idx}")
            print(e)

    batched_elements = [elements[i:i+BATCH_SIZE] for i in range(0, len(elements), BATCH_SIZE)][739:]

    print(f"Downloading {len(elements):,d} images in {len(batched_elements)}")

    download_images(batched_elements, workers=2)


    
    
        

    