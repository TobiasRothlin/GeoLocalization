import pandas as pd
import os
import shutil
import sys
import json
from tqdm import tqdm

OUTPUT_PATH = './dataset'

allCsvFilesInDir = [f for f in os.listdir('.') if f.endswith('.csv')]


# Load the data
data = pd.read_csv('./glare_train.csv')

print(data.head())

# Loop through the data
for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing data"):
    img_path = row['img_path']
    lat = row['lat']
    lon = row['lon']
    city = row['city']
    country = row['country']
    continent = row['continent']

    current_path = os.getcwd()

    # check if the image exists
    if os.path.exists(current_path +"/"+ img_path):

        # copy Image to the output path
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        if not os.path.exists(OUTPUT_PATH + "/" + img_path):
            shutil.copy(current_path +"/"+ img_path, OUTPUT_PATH + "/" + img_path.split("/")[-1])

            #remove the image from the current path
            os.remove(current_path +"/"+ img_path)
            # print("Copying image to: ", OUTPUT_PATH + "/" + img_path)

        json_path = OUTPUT_PATH + "/"+ img_path.split("/")[-1].replace("jpeg", "json")
        if not os.path.exists(json_path):
            # print("Creating json file for: ", img_path)
            json_data = {
                "img_path": img_path,
                "lat": lat,
                "lon": lon,
                "city": city,
                "country": country,
                "continent": continent
            }
            with open(json_path, 'w') as f:
                json.dump(json_data, f)
        else:
            print("Json file already exists for: ", img_path)
    else:
        print("Could not find img:" + current_path + img_path)