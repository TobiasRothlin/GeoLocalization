import os
import sys
sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from tqdm import tqdm

import json

from MapillaryCollection import MapillaryInterface

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")

DOWNLOAD_JSON_PATH = "/home/tobias.rothlin/GeoLocalization/src/DGX1/src/MapillaryCollection/download.json"

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
    print(f"Creating folder {new_path}")
    return new_path

if __name__ == "__main__":    
    with open("/home/tobias.rothlin/GeoLocalization/src/DGX1/src/MapillaryCollection/.apiTokenMapillary", "r") as f:
        api_key = f.read()


    with open(DOWNLOAD_JSON_PATH, "r") as f:
        download_data = json.load(f)

    mi = MapillaryInterface(api_key)

    print(f"Found {len(download_data['ToDownload'])} locations to download")

    for config in download_data["ToDownload"]:
        if config["DidDownload"]:
            print(f"Skipping {config['_country']} {config['lat']}, {config['lon']}")
            continue

        remove_cached_files()
        
        lat = config["lat"]
        lon = config["lon"]
        radius = config["radius"]
        max_images = config["numberOfImages"]
        zoom = config.get("zoom", 14)
        try:
            folder = get_unque_folder_name(f"Batch", TRAIN_DATA_FOLDER)
            number_of_locations_downloaded = mi.get_data(lat, lon, folder, radius=radius, max_images=max_images, zoom=zoom)
            config["DidDownload"] = True
            config["IamgesDownloaded"] = number_of_locations_downloaded
            
        except Exception as e:
            config["DidDownload"] = False
            config["IamgesDownloaded"] = 0
            print(f"Failed to download data for {config['_country']} {lat}, {lon}")
            print(e.with_traceback(sys.exc_info()[2]))
        
        with open(DOWNLOAD_JSON_PATH, "w") as f:
            json.dump(download_data, f, indent=4)

    