import pandas as pd
import os
import shutil
import sys
import json
from tqdm.notebook import tqdm

DATASET_FOLDER_INPUT = "./Dataset_OG"
OUTPUT_FOLDER = "./Dataset"
VALID_CSV_FILE_HEADER_REQUIREMENTS = ['img_path','city', 'country', 'continent', 'lat', 'lon']

def find_files(extension, folder):
    files_found = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                files_found.append(os.path.join(root, file))

    print(f"Found {len(files_found)} files with extension {extension}")
    return files_found


# Call the function for csv and jpg files
all_csv_files_in_folder = find_files('.csv', DATASET_FOLDER_INPUT)
all_images_in_folder = find_files('.jpg', DATASET_FOLDER_INPUT) + \
                       find_files('.jpeg', DATASET_FOLDER_INPUT) + \
                       find_files('.png', DATASET_FOLDER_INPUT)


all_images_in_folder[:5]


relative_path_images = ["img/" + file.split("/")[-2]+ "/" + file.split("/")[-1] for file in all_images_in_folder]
print(f"Trimmed {len(relative_path_images)}/{len(all_images_in_folder)} images")

rel_to_full_mapping = {rel: full for rel, full in zip(relative_path_images, all_images_in_folder)}

print(f"Mapping from relative to full path for images {len(rel_to_full_mapping)}")
rel_to_full_mapping


removed_doublicates = list(set(relative_path_images))
print(f"Removed {len(relative_path_images) - len(removed_doublicates)} duplicates")

valid_csv_files = []
invalid_csv_files = []
for file in all_csv_files_in_folder:
    with open(file, 'r') as f:
        header = f.readline().strip().split(',')
        for requirement in VALID_CSV_FILE_HEADER_REQUIREMENTS:
            if requirement not in header:
                invalid_csv_files.append(file)
                break
        else:
            valid_csv_files.append(file)

print(f"Found {len(valid_csv_files)} valid csv files")
print(f"Found {len(invalid_csv_files)} invalid csv files")

print(f"Invalid Files:")
for file in invalid_csv_files:
    print(file)


def process_img_data(img_path, city, country, continent, lat, lon):
    json_data = {
        "img_path": img_path,
        "city": city,
        "country": country,
        "continent": continent,
        "lat": lat,
        "lon": lon
    }

    # move image to output folder
    img_name = img_path.split("/")[-1]
    img_folder = "/".join(img_path.split("/")[:-2])
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    shutil.copyfile(img_path, os.path.join(OUTPUT_FOLDER, img_name))

    # save json data
    json_name = img_name.split(".")[0] + ".json"
    with open(os.path.join(OUTPUT_FOLDER,json_name), 'w') as f:
        json.dump(json_data, f)


processed_images = []
for csv_file in valid_csv_files:
    df = pd.read_csv(csv_file)
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {csv_file}"):
        rel_image_path = row['img_path']
        full_image_path = rel_to_full_mapping[rel_image_path]
        processed_images.append(full_image_path)
        process_img_data(full_image_path, row['city'], row['country'], row['continent'], row['lat'], row['lon'])


print(f"Processed {len(processed_images)} images")
print(f"Could not process {len(all_images_in_folder) - len(processed_images)} images")