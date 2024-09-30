from concurrent.futures import ThreadPoolExecutor, as_completed
from geopy.geocoders import Nominatim
import reverse_geocode
from functools import partial
import os
import json
import ssl
import certifi
import pandas as pd
import time
import socket 

from tqdm import tqdm

def is_connected():
    try:
        # Check if we can resolve the host name -- tells us if there is a DNS listening
        host = socket.gethostbyname("nominatim.openstreetmap.org")
        # Connect to the host -- tells us if the host is reachable
        s = socket.create_connection((host, 443), 2)
        s.close()
        return True
    except Exception:
        return False

def reverseGeoLocation(lat, lon):
    wasSuccessful = False
    MAX_RETRIES = 10

    location = None

    retries = 0
    while not wasSuccessful and MAX_RETRIES > retries:
        retries += 1
        if not is_connected():
            with open("error_log.txt", "a") as f:
                f.write("Network is unreachable. Retrying...\n")
            time.sleep(5)  # Wait before retrying
            continue

        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            geolocator = Nominatim(user_agent="geoapiExercises", ssl_context=ssl_context)
            reverse = partial(geolocator.reverse, language="en")
            loc = reverse(f"{lat}, {lon}")
            location = {
                "city": loc.raw["address"].get("city", None),
                "country": loc.raw["address"].get("country", None)
            }
            wasSuccessful = True

        except Exception as e:
            with(open("error_log.txt", "a")) as f:
                    f.write(f"Error in reverseGeoLocation: {e}\n")
            

    return location


def reverseGeoLocation_offline(lat, lon):
    res = reverse_geocode.get((lat, lon))
    location = {
                "city": res.get("city", None),
                "country": res.get("country", None)
            }
    return location



def put_location_to_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lat = data["lat"]
    lon = data["lon"]

    
    if "DidReverseGeoLocation" in data and data["DidReverseGeoLocation"]:
        return
    
    location = reverseGeoLocation_offline(lat, lon)

    if location is None:
        return
    data["city"] = location["city"]
    data["country"] = location["country"]
    data["DidReverseGeoLocation"] = True

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)

def put_locations_to_json_files(json_files, num_threads=8):
    with open("error_log.txt", "w") as f:
        f.write("")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(put_location_to_json, json_file): json_file for json_file in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Putting locations in JSON"):
            # Check for exceptions
            try:
                future.result()
            except Exception as e:
                with(open("error_log.txt", "a")) as f:
                    f.write(f"Error in {futures[future]}: {e}\n")
                continue



        