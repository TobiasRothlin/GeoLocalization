from concurrent.futures import ThreadPoolExecutor, as_completed
from geopy.geocoders import Nominatim
from functools import partial
import os
import json
import ssl
import certifi
import pandas as pd

from tqdm import tqdm

def get_position(json_file):
    with open(json_file, "r") as f:
        try:
            data = json.load(f)
        except UnboundLocalError:
            print(f"Error in {json_file}")
            return {"file": json_file, "lat": None, "lon": None}
        return {
            "file": json_file,
            "lat": data["lat"],
            "lon": data["lon"],
            "city": data.get("city", None),
            "country": data.get("country", None)
        }


def get_positions(json_files, num_threads=8):
    positions = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(get_position, json_file): json_file for json_file in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Getting positions"):
            positions.append(future.result())
    return positions

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


def reverseGeoLocation(lat, lon):
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    geolocator = Nominatim(user_agent="geoapiExercises", ssl_context=ssl_context)
    reverse = partial(geolocator.reverse, language="en")
    location = reverse(f"{lat}, {lon}")
    location = {
        "city": location.raw["address"].get("city", None),
        "country": location.raw["address"].get("country", None)
    }
    return location


def put_location_to_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lat = data["lat"]
    lon = data["lon"]

    
    if "DidReverseGeoLocation" in data and data["DidReverseGeoLocation"]:
        return
    
    location = reverseGeoLocation(lat, lon)
    data["city"] = location["city"]
    data["country"] = location["country"]
    data["DidReverseGeoLocation"] = True

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

def put_locations_to_json_files(json_files, num_threads=8):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(put_location_to_json, json_file): json_file for json_file in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Putting locations in JSON"):
            pass


def get_country_df():
    data = []
    with open("ListOfCountries.txt", "r") as f:
        raw_data = f.read()
        raw_data = raw_data.split("\n")
        for line in raw_data:
            line = line.split(";")
            data.append({
                "name": line[0],
                "code": line[1]
            })
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    print(reverseGeoLocation(37.7749, -122.4194))
    print(reverseGeoLocation(40.7128, -74.0060))
    print(reverseGeoLocation(51.5074, -0.1278))
    print(reverseGeoLocation(48.8566, 2.3522))
    print(reverseGeoLocation(55.7558, 37.6176))
    print(reverseGeoLocation(35.6895, 139.6917))
    print(reverseGeoLocation(37.5665, 126.9780))