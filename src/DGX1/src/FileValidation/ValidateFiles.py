from concurrent.futures import ThreadPoolExecutor, as_completed
from geopy.geocoders import Nominatim
from functools import partial
import os
import json
import ssl
import certifi
import pandas as pd

from time import sleep

from PIL import Image

from tqdm import tqdm

from PIL import Image


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


def find_dif_in_list(json_list, image_list):
    diffrences = []

    json_list = set(json_list)
    image_list = set(image_list)

    for json in tqdm(json_list, desc="Finding differences JSON"):
        image_path = json.replace('.json', '.jpg')
        if image_path not in image_list:
            if image_path not in diffrences:
                image_path = json.replace('.json', '.jpeg')
                if image_path not in image_list:
                    diffrences.append(json)

    for image in tqdm(image_list, desc="Finding differences Image"):
        json_path = image.replace('.jpg', '.json')
        if json_path not in json_list:
            if json_path not in diffrences:
                json_path = image.replace('.jpeg', '.json')
                if json_path not in json_list:
                    diffrences.append(image)

    return diffrences



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


def check_for_valid_image_file(image_path):
    try:
        with Image.open(image_path) as img:
            image = img.copy()
        image.convert("RGB")
        return True
    except:
        return False
    
def remove_invalid_image_file(image_path):
    if image_path.endswith(".jpg"):
        json_path = image_path.replace(".json", ".jpg")
    elif image_path.endswith(".jpeg"):
        json_path = image_path.replace(".json", ".jpeg")
    else:
        raise ValueError("Invalid image file extension")
    
    if os.path.exists(json_path) and os.path.exists(image_path):
        try:
            os.remove(json_path)
            os.remove(image_path)
        except Exception as e:
            print(f"    -! Could not remove {json_path} and {image_path} with error: {e}")
        print(f"Removed {json_path} and {image_path}")
    else:
        print(f"Could not remove {json_path} and {image_path}")

def check_for_valid_image_files(image_paths,remove_invalid=False,batch_id=0):
    checks = []
    for image in tqdm(image_paths, desc=f"Checking images in Batch {batch_id}"):
        if not check_for_valid_image_file(image):
            checks.append((False,image))
        else:
            checks.append((True,image))

    return checks

    
def check_for_valid_image_file_batched(image_paths, num_threads=16,remove_invalid=False):
    print(f"Checking {len(image_paths):,d} images")

    batchSize = len(image_paths) // (num_threads-1)
    image_batches = [image_paths[i:i + batchSize] for i in range(0, len(image_paths), batchSize)]

    invalid_files = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(check_for_valid_image_files, batch,remove_invalid,idx): batch for idx,batch in enumerate(image_batches)}
        checks = {key: 0 for key in ["Valid", "Invalid"]}
        for future in as_completed(futures):
            res = future.result()
            for check in res:
                if check[0]:
                    checks["Valid"] += 1
                else:
                    checks["Invalid"] += 1
                    invalid_files.append(check[1])

    for i in range(num_threads):
        print("")

    print("Invalid Files:")
    for file in invalid_files:
        print(f"  {file}")

    print("-"*100)
    print(f"Checked {len(image_paths):,d} images")
    for check in checks:
        print(" --|",end="")
        print_bar_chart(checks[check], len(image_paths) - checks[check], bar_length=100)
        print("|--",end="")
        print(f" {check}: {checks[check]:,d} ({(checks[check]/len(image_paths))*100:.2f}%)")


    if len(invalid_files) > 0:
        print("Removing invalid files...")
        answer = input("Do you want to remove the invalid files? (y/n): ")
        if answer.lower() == "y":
            for file in invalid_files:
                remove_invalid_image_file(file)
        else:
            print("No files removed")


    return checks
