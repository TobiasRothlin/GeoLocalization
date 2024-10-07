import mlflow
import os

from tqdm import tqdm

import PIL

import plotly.express as px

import plotly.graph_objects as go

import pandas as pd

import dotenv

import json

from concurrent.futures import ThreadPoolExecutor, as_completed

def check_file(json_path):
    checks = {
        "Valid Images Files": None,
        "Did Reverse Geocoding": None,
        "Did Classification": None,
        "Country": None,
        "City": None,
        "Region": None,
        "Population Area": None,
    }

    image_path = json_path.replace(".json", ".jpg")
    if not os.path.exists(image_path):
        image_path = json_path.replace(".json", ".jpeg")
        if not os.path.exists(image_path):
            checks["Valid Images Files"] = False
        
    try:
        PIL.Image.open(image_path)
        checks["Valid Images Files"] = True
    except:
        checks["Valid Images Files"] = False
    

    with open(json_path, "r") as f:
        data = json.load(f)
        checks["Did Reverse Geocoding"] = data.get("DidReverseGeoLocation", None) is not None
        checks["Did Classification"] = data.get("DidClassification", None) is not None

        if checks["Did Reverse Geocoding"]:
            checks["Country"] = data["country"]
            checks["City"] = data["city"]

        if checks["Did Classification"]:
            checks["Region"] = data["PredictedRegion"]
            checks["Population Area"] = data["PredictedPopulationArea"]
        
    return checks


def process_batch(batch_file):
    results = {
        "Valid Images Files": 0,
        "Did Reverse Geocoding": 0,
        "Did Classification": 0,
        "CountryCount": {},
        "CityCount": {},
        "RegionCount": {},
        "PopulationAreaCount": {},
    }

    for file in batch_file:
        checks = check_file(file)
        results["Valid Images Files"] += checks["Valid Images Files"]
        results["Did Reverse Geocoding"] += checks["Did Reverse Geocoding"]
        results["Did Classification"] += checks["Did Classification"]

        if checks["Country"] in results["CountryCount"]:
            results["CountryCount"][checks["Country"]] += 1
        else:
            results["CountryCount"][checks["Country"]] = 1

        if checks["City"] in results["CityCount"]:
            results["CityCount"][checks["City"]] += 1
        else:
            results["CityCount"][checks["City"]] = 1

        if checks["Region"] in results["RegionCount"]:
            results["RegionCount"][checks["Region"]] += 1
        else:
            results["RegionCount"][checks["Region"]] = 1

        if checks["Population Area"] in results["PopulationAreaCount"]:
            results["PopulationAreaCount"][checks["Population Area"]] += 1
        else:
            results["PopulationAreaCount"][checks["Population Area"]] = 1

    return results

def process_files(files, num_threads=64):
    batches = [files[i:i + num_threads] for i in range(0, len(files), num_threads)]
    results = {
        "Valid Images Files": 0,
        "Did Reverse Geocoding": 0,
        "Did Classification": 0,
        "CountryCount": {},
        "CityCount": {},
        "RegionCount": {},
        "PopulationAreaCount": {},
    }

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_batch, batch) for batch in tqdm(batches, desc="Starting Processing Files")]
        for future in tqdm(as_completed(futures), desc="Processing Files",total=len(futures)):
            try:
                result = future.result()
                results["Valid Images Files"] += result["Valid Images Files"]
                results["Did Reverse Geocoding"] += result["Did Reverse Geocoding"]
                results["Did Classification"] += result["Did Classification"]

                for country, count in result["CountryCount"].items():
                    if country in results["CountryCount"]:
                        results["CountryCount"][country] += count
                    else:
                        results["CountryCount"][country] = count

                for city, count in result["CityCount"].items():
                    if city in results["CityCount"]:
                        results["CityCount"][city] += count
                    else:
                        results["CityCount"][city] = count

                for region, count in result["RegionCount"].items():
                    if region in results["RegionCount"]:
                        results["RegionCount"][region] += count
                    else:
                        results["RegionCount"][region] = count

                for population_area, count in result["PopulationAreaCount"].items():
                    if population_area in results["PopulationAreaCount"]:
                        results["PopulationAreaCount"][population_area] += count
                    else:
                        results["PopulationAreaCount"][population_area] = count

                
            except Exception as e:
                print(f"Error in process_files: {e.with_traceback()}")
    
    return results


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


def sort_data(data):
    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=True)
    sorted_keys = [item[0] for item in sorted_items]
    sorted_values = [item[1] for item in sorted_items]
    return sorted_keys, sorted_values


def doVisualizaiton(trian_files,test_files):
    dotenv.load_dotenv(dotenv.find_dotenv())

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    print("Starting Visualization")

    mlflow.set_tracking_uri("https://mlflow.infs.ch")
    mlflow.set_experiment("GeoLocalization_Data_Visualization")
    

    with mlflow.start_run():
        # Pi Chart showing the distribution of the data
        fig = px.pie(values=[len(trian_files), len(test_files)], names=["Train", "Test"], title="Data Distribution")
        mlflow.log_figure(fig, "DataDistribution.html")

        plot_data = {
        }

        plot_data["Train"] = process_files(trian_files)
        plot_data["Test"] = process_files(test_files)

        mlflow.log_param("Train_ValidImagesFiles", plot_data["Train"]["Valid Images Files"])
        mlflow.log_param("Train_InvalidImagesFiles", len(trian_files)-plot_data["Train"]["Valid Images Files"])

        # Pi Chart showing how many files are valid
        fig = px.pie(values=[plot_data["Train"]["Valid Images Files"], len(trian_files)-plot_data["Train"]["Valid Images Files"]], names=["Valid", "Invalid"], title="Valid Images Files")
        mlflow.log_figure(fig, "Train_ValidImagesFiles.html")

        fig = px.pie(values=[plot_data["Test"]["Valid Images Files"], len(test_files)-plot_data["Test"]["Valid Images Files"]], names=["Valid", "Invalid"], title="Valid Images Files")
        mlflow.log_figure(fig, "Test_ValidImagesFiles.html")

        # Pi Chart showing how many files are reverse geocoded
        fig = px.pie(values=[plot_data["Train"]["Did Reverse Geocoding"], len(trian_files)-plot_data["Train"]["Did Reverse Geocoding"]], names=["Reverse Geocoded", "Not Reverse Geocoded"], title="Reverse Geocoded Files")
        mlflow.log_figure(fig, "Train_ReverseGeocodedFiles.html")

        fig = px.pie(values=[plot_data["Test"]["Did Reverse Geocoding"], len(test_files)-plot_data["Test"]["Did Reverse Geocoding"]], names=["Reverse Geocoded", "Not Reverse Geocoded"], title="Reverse Geocoded Files")
        mlflow.log_figure(fig, "Test_ReverseGeocodedFiles.html")

        # Pi Chart showing how many files are classified
        fig = px.pie(values=[plot_data["Train"]["Did Classification"], len(trian_files)-plot_data["Train"]["Did Classification"]], names=["Classified", "Not Classified"], title="Classified Files")
        mlflow.log_figure(fig, "Train_ClassifiedFiles.html")

        fig = px.pie(values=[plot_data["Test"]["Did Classification"], len(test_files)-plot_data["Test"]["Did Classification"]], names=["Classified", "Not Classified"], title="Classified Files")
        mlflow.log_figure(fig, "Test_ClassifiedFiles.html")

        # Bar Chart showing the distribution of the countries
        train_country_keys, train_country_values = sort_data(plot_data["Train"]["CountryCount"])
        fig = px.bar(x=train_country_keys, y=train_country_values, title="Countries Distribution")
        mlflow.log_figure(fig, "Train_CountriesDistribution.html")

        test_country_keys, test_country_values = sort_data(plot_data["Test"]["CountryCount"])
        fig = px.bar(x=test_country_keys, y=test_country_values, title="Countries Distribution")
        mlflow.log_figure(fig, "Test_CountriesDistribution.html")

        # Bar Chart showing the distribution of the cities
        train_city_keys, train_city_values = sort_data(plot_data["Train"]["CityCount"])
        fig = px.bar(x=train_city_keys, y=train_city_values, title="Cities Distribution")
        mlflow.log_figure(fig, "Train_CitiesDistribution.html")

        test_city_keys, test_city_values = sort_data(plot_data["Test"]["CityCount"])
        fig = px.bar(x=test_city_keys, y=test_city_values, title="Cities Distribution")
        mlflow.log_figure(fig, "Test_CitiesDistribution.html")

        # Bar Chart showing the distribution of the regions
        train_region_keys, train_region_values = sort_data(plot_data["Train"]["RegionCount"])
        fig = px.bar(x=train_region_keys, y=train_region_values, title="Regions Distribution")
        mlflow.log_figure(fig, "Train_RegionsDistribution.html")

        test_region_keys, test_region_values = sort_data(plot_data["Test"]["RegionCount"])
        fig = px.bar(x=test_region_keys, y=test_region_values, title="Regions Distribution")
        mlflow.log_figure(fig, "Test_RegionsDistribution.html")

        # Bar Chart showing the distribution of the population areas
        train_population_keys, train_population_values = sort_data(plot_data["Train"]["PopulationAreaCount"])
        fig = px.bar(x=train_population_keys, y=train_population_values, title="Population Area Distribution")
        mlflow.log_figure(fig, "Train_PopulationAreaDistribution.html")

        test_population_keys, test_population_values = sort_data(plot_data["Test"]["PopulationAreaCount"])
        fig = px.bar(x=test_population_keys, y=test_population_values, title="Population Area Distribution")
        mlflow.log_figure(fig, "Test_PopulationAreaDistribution.html")


        # Scatter Map showing the distribution of the data
        train_latlong = getLatLongFiles(trian_files)
        test_latlong = getLatLongFiles(test_files)

        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lon=[x[1] for x in train_latlong],
            lat=[x[0] for x in train_latlong],
            mode='markers',
            marker=dict(
                size=8,
                opacity=0.8,
                reversescale=True,
                autocolorscale=False,
                symbol='circle',
                line=dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                ),
                colorscale='Blues',
                cmin=0,
                color=[],
                cmax=0,
                colorbar=dict(
                    title="Train Data"
                )
            )
        ))

        mlflow.log_figure(fig, "Train_DataDistribution.html")


        fig = go.Figure()
        fig.add_trace(go.Scattergeo(
            lon=[x[1] for x in test_latlong],
            lat=[x[0] for x in test_latlong],
            mode='markers',
            marker=dict(
                size=8,
                opacity=0.8,
                reversescale=True,
                autocolorscale=False,
                symbol='circle',
                line=dict(
                    width=1,
                    color='rgba(102, 102, 102)'
                ),
                colorscale='Blues',
                cmin=0,
                color=[],
                cmax=0,
                colorbar=dict(
                    title="Test Data"
                )
            )
        ))

        mlflow.log_figure(fig, "Test_DataDistribution.html")


        # Heat Map showing the distribution of the data
        fig = go.Figure()
        fig.add_trace(go.Densitymapbox(lat=[x[0] for x in train_latlong], lon=[x[1] for x in train_latlong], radius=10))
        fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=0)
        mlflow.log_figure(fig, "Train_HeatMap.html")

        fig = go.Figure()
        fig.add_trace(go.Densitymapbox(lat=[x[0] for x in test_latlong], lon=[x[1] for x in test_latlong], radius=10))
        fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=0)
        mlflow.log_figure(fig, "Test_HeatMap.html")







