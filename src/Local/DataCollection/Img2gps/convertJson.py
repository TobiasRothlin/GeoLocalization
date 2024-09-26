import json
import os
import sys
from tqdm import tqdm


def find_files(extension, folder):
    files_found = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(extension):
                files_found.append(os.path.join(root, file))

    print(f"Found {len(files_found)} files with extension {extension}")
    return files_found



all_json_files = find_files(".json", "./")

for file in tqdm(all_json_files):
    with open(file, "r") as f:
        try:
            original_data = json.load(f)
        except:
            print(f"Error reading file {file}")
            continue

    if "Comment" not in original_data:
        print(f"File {file} does not have Comment key")
        continue

    if len(original_data["Comment"]) < 10:
        print(f"File {file} does not have enough Comment keys")
        continue

    new_json_data = {
        "img_path": file.replace(".json", ".jpg"),
        "city": "",
        "country": "",
        "continent": "",
        "lat": float(original_data["Comment"][8].split(":")[1].strip()),
        "lon": float(original_data["Comment"][9].split(":")[1].strip())
    }

    with open(file, "w") as f:
        json.dump(new_json_data, f, indent=4)