import pandas as pd
import os
import shutil
import sys
import json
from tqdm.notebook import tqdm

FOLDER_PATH = "./"

OUTPUT_PATH = "./output/"

def find_files(extension, folder):
    files_found = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                files_found.append(os.path.join(root, file))

    print(f"Found {len(files_found)} files with extension {extension}")
    return files_found


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


all_csv_files = find_files(".csv", FOLDER_PATH)
all_jpg_files = find_files(".jpg", FOLDER_PATH)


all_raw_csv_files = [file for file in all_csv_files if "raw" in file]
print(len(all_raw_csv_files))

for file in all_raw_csv_files[:5]:
    print(file)
    

print(100*"-")

for file in all_jpg_files[:5]:
    print(file)


processes_jpg = []

def process_csv_file(file):
    df = pd.read_csv(file)
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {file}"):
        image_name = row["key"]
        lat = row["lat"]
        lon = row["lon"]

        matched_image = None
        for jpg_file in all_jpg_files:
            if image_name in jpg_file:
                matched_image = jpg_file
                break

        if matched_image is not None:
            json_data = {
                        "img_path": "",
                        "city": "",
                        "country": "",
                        "continent": "",
                        "lat": lat,
                        "lon": lon
                    } 

            if os.path.exists(OUTPUT_PATH + image_name + ".json"):
                print(f"File {OUTPUT_PATH + image_name + '.json'} already exists. Skipping")
                
            else:    
                json.dump(json_data, open(OUTPUT_PATH + image_name + ".json", "w")) 

            if os.path.exists(OUTPUT_PATH + image_name + ".jpg"):
                print(f"File {OUTPUT_PATH + image_name + '.jpg'} already exists. Skipping")
            else:
                shutil.copy(matched_image, OUTPUT_PATH + image_name + ".jpg")

            processes_jpg.append(matched_image)
        else:
            print(f"Could not find image for {image_name}")


for file in all_raw_csv_files:
    process_csv_file(file)

print(f"Processed {len(all_raw_csv_files)} files")
print(f"Processed {len(processes_jpg)} images")
print(f"Processed {len(all_jpg_files)} images")