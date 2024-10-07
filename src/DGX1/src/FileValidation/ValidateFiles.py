from concurrent.futures import ThreadPoolExecutor, as_completed
from geopy.geocoders import Nominatim
from functools import partial
import os
import json
import ssl
import certifi
import pandas as pd

from PIL import Image

from tqdm import tqdm


def print_bar_chart(a, b, bar_length=50,color_a = "\033[32m", color_b = "\033[31m", reset_color = "\033[0m"):
    total = a + b
    if total == 0:
        print("No data to display.")
        return

    # Calculate the proportion of each color
    a_length = int((a / total) * bar_length)
    b_length = bar_length - a_length

    # Create the bar
    bar = f"{color_a}{'█' * a_length}{color_b}{'█' * b_length}{reset_color}"

    # Print the bar
    print(bar,end="")



def check_single_file(json_file, required_keys=None):
    checkes = {}

    if required_keys is None:
        required_keys = ["city", "country", "lat", "lon", "PredictedPopulationArea", "PredictedRegion", "DidReverseGeoLocation", "DidClassification"]
    with open(json_file, "r") as f:
        data = json.load(f)
    for key in required_keys:
        checkes[key] = key in data


    image_path = json_file.replace(".json", ".jpg")
    if not os.path.exists(image_path):
        image_path = json_file.replace(".json", ".jpeg")
        if not os.path.exists(image_path):
            checkes["ImageExists"] = False
        else:
            checkes["ImageExists"] = True

    else:
        checkes["ImageExists"] = True


    try:
        image = Image.open(image_path)
        checkes["CanOpenImage"] = True
    except:
        checkes["CanOpenImage"] = False

    return checkes


def check_batch_of_files(json_batch):
    checks = {
        "city": 0,
        "country": 0,
        "lat": 0,
        "lon": 0,
        "PredictedPopulationArea": 0,
        "PredictedRegion": 0,
        "DidReverseGeoLocation": 0,
        "DidClassification": 0,
        "ImageExists": 0,
        "CanOpenImage": 0
    }

    for file in json_batch:
        res = check_single_file(file)
        for key in res:
            if res[key]:
                checks[key] += 1

    return checks


def check_json_files(json_files, num_threads=16):
    print(f"Checking {len(json_files):,d} files")

    json_batches = [json_files[i:i + num_threads] for i in range(0, len(json_files), num_threads)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(check_batch_of_files, batch): batch for batch in json_batches}
        checks = {key: 0 for key in ["city", "country", "lat", "lon", "PredictedPopulationArea", "PredictedRegion", "DidReverseGeoLocation", "DidClassification", "ImageExists", "CanOpenImage"]}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Checking files"):
            res = future.result()
            for key in res:
                checks[key] += res[key]

    print(f"Checked {len(json_files):,d} files")
    for check in checks:
        print(" --|",end="")
        print_bar_chart(checks[check], len(json_files) - checks[check], bar_length=100)
        print("|--",end="")
        print(f" {check}: {checks[check]:,d} ({(checks[check]/len(json_files))*100:.2f}%)")
    print("")

    return checks