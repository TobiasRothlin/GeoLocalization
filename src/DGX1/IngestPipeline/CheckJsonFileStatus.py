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


def check_json_files(json_files, num_threads=16):
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

    print(f"Checking {len(json_files):,d} files")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(check_single_file, json_file) for json_file in json_files]
        for future in tqdm(as_completed(futures), total=len(json_files), desc=f"Checking JSON Files"):
            result = future.result()
            for key in result:
                if result[key]:
                    checks[key] += 1


    print(f"Checked {len(json_files):,d} files")
    for check in checks:
        print(" --|",end="")
        print_bar_chart(checks[check], len(json_files) - checks[check], bar_length=100)
        print("|--",end="")
        print(f" {check}: {checks[check]:,d} ({(checks[check]/len(json_files))*100:.2f}%)")
    print("")