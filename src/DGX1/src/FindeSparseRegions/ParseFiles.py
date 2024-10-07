import os

from tqdm import tqdm

import json

from concurrent.futures import ThreadPoolExecutor, as_completed

def getLatLong(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        return data["lat"], data["lon"]
    
def getLatLongBatch(batch_file):
    results = []
    for file in batch_file:
        results.append(getLatLong(file))
    return results

def getLatLongFiles(files, num_threads=64):
    batches = [files[i:i + num_threads] for i in range(0, len(files), num_threads)]
    results = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(getLatLongBatch, batch) for batch in tqdm(batches, desc="Starting Getting LatLong")]
        for future in tqdm(as_completed(futures), desc="Getting LatLong" ,total=len(futures)):
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                print(f"Error in getLatLongFiles: {e.with_traceback()}")
    
    return results


def getCountries(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        return data["country"]
    
def getCountriesBatch(batch_file):
    results = []
    for file in batch_file:
        results.append(getCountries(file))
    return results

def getCountriesFiles(files, num_threads=64):
    batches = [files[i:i + num_threads] for i in range(0, len(files), num_threads)]
    results = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(getCountriesBatch, batch) for batch in tqdm(batches, desc="Starting Getting Countries")]
        for future in tqdm(as_completed(futures), desc="Getting Countries" ,total=len(futures)):
            try:
                result = future.result()
                results.extend(result)
            except Exception as e:
                print(f"Error in getCountriesFiles: {e.with_traceback()}")
    
    return results