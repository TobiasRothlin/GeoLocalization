from concurrent.futures import ThreadPoolExecutor, as_completed

import os

from tqdm import tqdm

import json

from time import time

import pandas as pd


def get_data_from_file(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    image_path = json_file.replace(".json", ".jpg")

    if not os.path.exists(image_path):
        image_path = json_file.replace(".json", ".jpeg")
        if not os.path.exists(image_path):
            print(f"Image file {image_path} not found")
            return None

    clean_data = {
        "imgPath": image_path,
        "jsonPath": json_file,
        "city": data.get("city", None),
        "country": data.get("country", None),
        "lat": data.get("lat", None),
        "lon": data.get("lon", None),
        "PopulationAreas": data.get("PopulationAreas", None),
        "Regions": data.get("Regions", None),
        "DidReverseGeoLocation": data.get("DidReverseGeoLocation", False),
        "DidClassification": data.get("DidClassification", False)
    }
    return clean_data



def createLableFile(json_files, output_file, number_of_threads=16):
    data = []

    with ThreadPoolExecutor(max_workers=number_of_threads) as executor:
        futures = [executor.submit(get_data_from_file, json_file) for json_file in json_files]
        for future in tqdm(as_completed(futures), total=len(json_files), desc=f"Creating Label File"):
            result = future.result()
            if result:
                data.append(result)

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=True)
    print(f"Saved data to {output_file}")