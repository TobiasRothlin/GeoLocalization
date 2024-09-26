import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

from plotly.subplots import make_subplots

import random
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import sys
import json

from tqdm import tqdm
import emoji

from main_utility import get_positions , print_bar_chart, put_locations_to_json_files

from DataLocator import DataLocator

TEST_DATA_FOLDER = "/Volumes/DATASET/Test"
TRAIN_DATA_FOLDER = "/Volumes/DATASET/Train"

MAKE_PLOTS = True
RUN_REVERSE_GEOCODING = False
USE_CASHED_DATA_FRAME = True

def setup():
    with open("Dataset/.apiTokenMapBox","r") as f:
        MAPBOX_TOKEN = f.read().strip()

    px.set_mapbox_access_token(MAPBOX_TOKEN)

    print("Setup Complete")




def main():
    setup()
    # Get the list of files in the test and train folders
    print("\033[37m") # Light Gray
    dl_Test = DataLocator(TEST_DATA_FOLDER)
    dl_Train = DataLocator(TRAIN_DATA_FOLDER)

    test_files_json = dl_Test.get_files(".json")
    test_files_jpg = dl_Test.get_files(".jpg")
    test_files_jpeg = dl_Test.get_files(".jpeg")

    train_files_json = dl_Train.get_files(".json")
    train_files_jpg = dl_Train.get_files(".jpg")
    train_files_jpeg = dl_Train.get_files(".jpeg")
    print("\033[0m") # Reset Color

    print(f"Checking if Meta Data files are present")
    test_files_json_meta = [f for f in test_files_json if "._" in f]
    test_files_jpg_meta = [f for f in test_files_jpg if "._" in f]
    test_files_jpeg_meta = [f for f in test_files_jpeg if "._" in f]

    train_files_json_meta = [f for f in train_files_json if "._" in f]
    train_files_jpg_meta = [f for f in train_files_jpg if "._" in f]
    train_files_jpeg_meta = [f for f in train_files_jpeg if "._" in f]

    if len(test_files_json_meta) == 0 and len(test_files_jpg_meta) == 0 and len(test_files_jpeg_meta) == 0:
        print("✅ \033[32m No Meta Data files found in Test folder \033[0m")

    else:
        print(f"❌ \033[31m {len(test_files_json_meta)+len(test_files_jpg_meta)+len(test_files_jpeg_meta)} Meta Data files found in Test folder \033[0m")

        if input("Do you want to remove these files? (y/n): ").lower() == "y":
            for f in test_files_json_meta + test_files_jpg_meta + test_files_jpeg_meta:
                os.remove(f)
                print(f"    -Removed {f}")


    if len(train_files_json_meta) == 0 and len(train_files_jpg_meta) == 0 and len(train_files_jpeg_meta) == 0:
        print("✅ \033[32m No Meta Data files found in Train folder \033[0m")
    
    else:
        print(f"❌ \033[31m {len(train_files_json_meta)+len(train_files_jpg_meta)+len(train_files_jpeg_meta)} Meta Data files found in Train folder \033[0m")

        if input("Do you want to remove these files? (y/n): ").lower() == "y":
            for f in train_files_json_meta + train_files_jpg_meta + train_files_jpeg_meta:
                os.remove(f)
                print(f"    -Removed {f}")

    

    print("\033[0m") # Reset Color

    print(f"\033[1m\u001b[32;1mTrain:\033[0m") # Bold and Green
    print(f"    JSON: {len(train_files_json):,d}")
    print(f"    JPG : {len(train_files_jpg):,d}")
    print(f"    JPEG: {len(train_files_jpeg):,d}")
    print(f"        Total: {len(train_files_jpg) +len(train_files_jpeg):,d}")

    if len(train_files_json) == len(train_files_jpg) +len(train_files_jpeg):
        print("✅ \033[32m All JSON and Image files are present in the Train folder \033[0m") # Green
    else:
        print(f"❌ \033[31m {len(train_files_json)-(len(train_files_jpg) +len(train_files_jpeg))} are missing ! \033[0m")   # Red

    print("\033[37m")   # Yellow
    print(100*"-")
    print("\033[0m")    # Reset Color



    print(f"\033[1m\u001b[34;1mTest:\033[0m")   # Bold
    print(f"    JSON: {len(test_files_json):,d}")
    print(f"    JPG : {len(test_files_jpg):,d}")
    print(f"    JPEG: {len(test_files_jpeg):,d}")
    print(f"        Total: {len(test_files_jpg) +len(test_files_jpeg):,d}")

    if len(test_files_json) == len(test_files_jpg) +len(test_files_jpeg):
        print("✅ \033[32m All JSON and Image files are present in the Test folder \033[0m")    # Green
    else:
        print(f"❌ \033[31m {len(test_files_json)-(len(test_files_jpg) +len(test_files_jpeg))} are missing ! \033[0m")  # Red
    
    print("\n")
    print("Data Distribution:")
    print_bar_chart(len(train_files_json), len(test_files_json), bar_length=100 ,color_a="\u001b[32;1m", color_b="\u001b[34;1m")    # Green and Blue
    print(f"    \u001b[32;1mTrain:{(100*len(train_files_json))/(len(train_files_json)+len(test_files_json)):.3f}% \u001b[34;1mTest:{(100*len(test_files_json))/(len(train_files_json)+len(test_files_json)):.3f}%")
    print("\n")

    
    if RUN_REVERSE_GEOCODING:
        print("\033[37m")
        put_locations_to_json_files(train_files_json, num_threads=16)
        put_locations_to_json_files(test_files_json, num_threads=16)
        print("\033[0m")


    if USE_CASHED_DATA_FRAME:
        train_df = pd.read_orc("train.orc")
        test_df = pd.read_orc("test.orc")

    else:
        print("\033[37m") # Light Gray
        train_positions = get_positions(train_files_json, num_threads=32)
        test_positions = get_positions(test_files_json, num_threads=16)
        print("\033[0m") # Reset Color

        train_df = pd.DataFrame(train_positions)
        test_df = pd.DataFrame(test_positions)

        train_df.to_orc("train.orc")
        test_df.to_orc("test.orc")

    if MAKE_PLOTS:
        print("\033[1m\u001b[32;1mTrain:\033[0m") # Bold and Green
        print(train_df.describe())
        print("\033[1m\u001b[34;1mTest:\033[0m") # Bold and Blue
        print(test_df.describe())
        
        #make heat map plot with the training data
        fig = go.Figure(go.Densitymap(lat=train_df["lat"], lon=train_df["lon"], radius=10))
        fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=0)
        fig.update_layout(map_style="open-street-map", map_center_lon=180)
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.show()


        

        #make bar char by training data per country using plotly
        train_df = train_df.dropna()
        
        px.bar(train_df["country"].value_counts(), title="Train Data per Country").show()

        


    



if __name__ == "__main__":
    main()
    