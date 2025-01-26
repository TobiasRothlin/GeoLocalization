from concurrent.futures import ThreadPoolExecutor, as_completed
from geopy.geocoders import Nominatim
import pycountry_convert as pc
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

country_mapping ={
    "Libyan Arab Jamahiriya": "Libya",
    "Palestinian Territory": "Palestine",
    "Korea, Democratic People's Republic of": "North Korea",

}

alpha2_mapping = {
    "VA": "IT",
}

country_to_continent_mapping = {
    "Kosovo": "Europe",
    "Reunion": "Africa",
    "Sint Maarten": "North America",
    "Bonaire, Saint Eustatius and Saba": "North America",
    "Saint Bartelemey": "North America",
    "Curacao": "North America",
    "Aland Islands": "Europe",
    "Cote d'Ivoire": "Africa",
    "Timor-Leste": "Asia",
    "Saint Helena": "Africa",
    "Western Sahara": "Africa",
}


def country_to_continent(country_name,lat=None,lon=None):
    if country_name is None:
        return None
    try:
        if country_name in country_to_continent_mapping:
            return country_to_continent_mapping[country_name]

        country_alpha2 = pc.country_name_to_country_alpha2(country_name)

        if country_alpha2 in alpha2_mapping:
            country_alpha2 = alpha2_mapping[country_alpha2]

        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    except Exception as e:
        with open("error_log.txt", "a") as f:
            f.write(f"Error in country_to_continent: {e}, lat={lat},lon={lon},country_name=({country_name})\n")
        country_continent_name = None

    return country_continent_name


def reverseGeoLocation_offline(lat, lon):
    res = reverse_geocode.get((lat, lon))
    country_name = res.get("country", None)
    if country_name in country_mapping:
        country_name = country_mapping[country_name]
    location = {
        "city": res.get("city", None),
        "country": country_name,
        "continent": country_to_continent(country_name,lat,lon),
    }
    return location

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

def put_location_to_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    lat = data["lat"]
    lon = data["lon"]

    
    if "DidReverseGeoLocation" in data and data["DidReverseGeoLocation"]:
        pass
    
    location = reverseGeoLocation_offline(lat, lon)

    if location is None:
        return
    if "city" not in data:
        data["city"] = None
    if "country" not in data:
        data["country"] = None
    if "continent" not in data:
        data["continent"] = None

    if data["city"] is None:
        data["city"] = location["city"]

    if data["country"] is None:
        data["country"] = location["country"]

    if data["continent"] is None:
        data["continent"] = location["continent"]

    if data["continent"] == "":
        if not location["continent"] == "":
            data["continent"] = location["continent"]
        else:
            raise ValueError(f"Continent is empty: {json_file} in file: {json_file}")
    
    if data["country"] == "":
        if not location["country"] == "":
            data["country"] = location["country"]
        else:
            raise ValueError(f"Country is empty: {json_file} in file: {json_file}")
    
    if data["city"] == "":
        if not location["city"] == "":
            data["city"] = location["city"]
        else:
            raise ValueError(f"City is empty: {json_file} in file: {json_file}")

    data["DidReverseGeoLocation"] = True

    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


def put_locations_to_json_batches(batch):
    for json_file in batch:
        put_location_to_json(json_file)

def put_locations_to_json_files(json_files, num_threads=8,error_log="error_log.txt"):
    with open(error_log, "w") as f:
        f.write("")

    batches = [json_files[i:i + num_threads] for i in range(0, len(json_files), num_threads)]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(put_locations_to_json_batches, batch) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures),desc="Reverse Geo Coding"):
            try:
                future.result()
            except Exception as e:
                with open(error_log, "a") as f:
                    f.write(f"Error in put_locations_to_json_files: {e}\n")

        print("Done Reverse Geo Coding")


        