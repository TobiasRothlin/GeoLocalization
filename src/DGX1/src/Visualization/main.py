import os
import sys

from tqdm import tqdm
import json
import random
import plotly.graph_objects as go

sys.path.append('/home/tobias.rothlin/GeoLocalization/src/DGX1/src/Utility')

from time import sleep

from DataLocator import DataLocator

from Visualization import doVisualizaiton

BASE_PATH = "/home/tobias.rothlin/data/GeoDataset"

TEST_DATA_FOLDER = os.path.join(BASE_PATH, "Test")
TRAIN_DATA_FOLDER = os.path.join(BASE_PATH, "Train")

def get_lat_lon_from_json_files(files):
    lat = []
    lon = []
    for file in tqdm(files,desc="Reading JSON Files"):
        with open(file) as f:
            data = json.load(f)
            lat.append(data["lat"])
            lon.append(data["lon"])
    return lat,lon

def do_geo_scatter_plot(lat,lon):
    fig = go.Figure(data=go.Scattergeo(
                lon = lon,
                lat = lat,
                mode = 'markers',
                marker_color = "blue",
            ))
    fig.update_geos(projection_type="natural earth")
    fig.update_layout(title=f"Holdout Test Data Distribution")
    fig.write_html("./GeoDataset2024_Train_Data_Distribution.html")

if __name__ == "__main__":

    dl_Subset = DataLocator(TRAIN_DATA_FOLDER,use_cache=False)

    files = dl_Subset.get_files(".json")

    random.shuffle(files)

    lat,lon = get_lat_lon_from_json_files(files[:5000])

    do_geo_scatter_plot(lat,lon)

    raise Exception("Stop here")

    print("\033[37m") # Light Gray
    
    dl_Test = DataLocator(TEST_DATA_FOLDER,use_cache=False)
    dl_Train = DataLocator(TRAIN_DATA_FOLDER,use_cache=False)

    test_files_json = dl_Test.get_files(".json")

    train_files_json = dl_Train.get_files(".json")

    print("\033[0m") # Reset Color

    
    doVisualizaiton(train_files_json,test_files_json)
    