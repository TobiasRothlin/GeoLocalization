import os

import requests
import mercantile
import json

from tqdm import tqdm

import random

import numpy as np

import plotly.express as px



class MapillaryInterface:

    def __init__(self, api_key, error_log="error.txt"):
        """
        Initialize the Mapillary Interface
        :param api_key: Mapillary API Key
        :param error_log: Path to the error log file
        """
        self.access_token = api_key
        self.error_log = error_log

        # Create the error log file if it does not exist
        with open(self.error_log, "w") as f:
            pass

    def __haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points
        :param lat1: Latitude of the first point
        :param lon1: Longitude of the first point
        :param lat2: Latitude of the second point
        :param lon2: Longitude of the second point
        :return: Haversine distance
        """
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(
            dlon / 2) * np.sin(dlon / 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c




    def __get_bbox(self, center_lat, center_lon, radius_km=100,show_plot=False):
        """
        Get the bounding box for a given center point and radius
        :param center_lat: Latitude of the center point
        :param center_lon: Longitude of the center point
        :param radius_km: Radius in km
        :return: Tuple of (lat1, lon1, lat2, lon2) representing the bounding box
        """
        if not (-180 <= center_lon <= 180 and -90 <= center_lat <= 90):
            raise ValueError("Latitude and Longitude are out of range.")

        R = 6371  # Earth radius in km

        lat1 = center_lat - np.degrees(radius_km / R)
        lat2 = center_lat + np.degrees(radius_km / R)
        lon1 = center_lon - np.degrees(radius_km / R / np.cos(np.radians(center_lat)))
        lon2 = center_lon + np.degrees(radius_km / R / np.cos(np.radians(center_lat)))

        print(f"Distance between center and corners: {self.__haversine_distance(center_lat, center_lon, lat1, lon1)} km")
        print(f"Distance between center and corners: {self.__haversine_distance(center_lat, center_lon, lat1, lon2)} km")

        if show_plot:
            fig = px.scatter_geo()
            fig.add_trace(px.scatter_geo(lat=[center_lat], lon=[center_lon], color=[0]).data[0])
            fig.add_trace(px.scatter_geo(lat=[lat1, lat2], lon=[lon1, lon2], color=[1,2]).data[0])
            fig.show()

        return lat1, lon1, lat2, lon2

    def __get_image_data_by_bbox(self, lat1, lon1, lat2, lon2,zoom=14,show_plot=False):
        """
        Get image data by bounding box
        :param lat1: Latitude of the first point
        :param lon1: Longitude of the first point
        :param lat2: Latitude of the second point
        :param lon2: Longitude of the second point
        :param zoom: Zoom level
        :return: List of image data
        """
        tiles = list(mercantile.tiles(lon1, lat1, lon2, lat2, zoom))
        bbox_list = [mercantile.bounds(tile.x, tile.y, tile.z) for tile in tiles]
        print(f"Found {len(bbox_list)} tiles")
        lat_min = min([bbox.south for bbox in bbox_list])
        lat_max = max([bbox.north for bbox in bbox_list])
        lon_min = min([bbox.west for bbox in bbox_list])
        lon_max = max([bbox.east for bbox in bbox_list])
        print(f"Requested Bounding Box: {lat1},{lon1},{lat2},{lon2}")
        print(f"Actual Bounding Box: {lat_min},{lon_min},{lat_max},{lon_max}")



        if show_plot:

            fig = px.scatter_geo()
            for i,tile in enumerate(tiles):
                lon, lat = mercantile.ul(tile.x, tile.y, tile.z)
                fig.add_trace(px.scatter_geo(lat=[lat], lon=[lon], color=[i+1]).data[0])

            fig.show()


        image_data = []

        progress_bar = tqdm(desc="Downloading Image Data", unit=" chunks", total=len(bbox_list))

        for bbox in bbox_list:
            bbox_str = str(f'{bbox.west},{bbox.south},{bbox.east},{bbox.north}')

            if not (
                    -180 <= bbox.west <= 180 and -180 <= bbox.east <= 180 and -90 <= bbox.south <= 90 and -90 <= bbox.north <= 90):
                with open(self.error_log, "a") as f:
                    f.write(f"--Invalid Bounding Box: {bbox_str}\n")

            fields = [
                "thumb_original_url",
                "geometry",
                "is_pano"
            ]
            # Rate limit for images? is 10'000 per minute
            url = f'https://graph.mapillary.com/images?access_token={self.access_token}&bbox={bbox_str}&fields={",".join(fields)}'
            while url:
                response = requests.get(url)
                data = response.json()
                if data:
                    try:
                        image_data += data["data"]
                    except KeyError as e:
                        with open(self.error_log, "a") as f:
                            f.write(f"--KeyError: {e} in {data}\n")

                    url = data.get('paging', {}).get('next', None)
            progress_bar.update(1)
            progress_bar.set_postfix({"Found Images": len(image_data)})

        progress_bar.close()
        return image_data

    def __is_image_data_valid(self, image_data):
        """
        Check if the image data is valid
        :param image_data: Image data
        :return: True if valid, False otherwise
        """
        if "thumb_original_url" not in image_data:
            return False

        if "geometry" not in image_data:
            return False

        if "coordinates" not in image_data["geometry"]:
            return False

        if len(image_data["geometry"]["coordinates"]) != 2:
            return False

        if "is_pano" not in image_data:
            return False

        return True

    def __get_image_by_url(self, url, file_name):
        """
        Get the image by URL
        :param url: URL of the image
        :param file_name: File name to save the image
        """
        response = requests.get(url)
        with open(file_name, "wb") as f:
            f.write(response.content)

    def __process_image_data(self, image_data):
        """
        Process the image data to remove duplicates
        :param image_data: List of image data
        :return: List of processed image data
        """
        clean_data = []
        processed_thumbs = set([])
        for sample in tqdm(image_data, desc="Processing Image Data"):
            if not self.__is_image_data_valid(sample):
                with open(self.error_log, "a") as f:
                    f.write(f"--Invalid Image Data: {sample}\n")
            else:
                if sample["thumb_original_url"] not in processed_thumbs:
                    clean_data.append({
                        "thumb_original_url": sample["thumb_original_url"],
                        "lat": sample["geometry"]["coordinates"][1],
                        "lon": sample["geometry"]["coordinates"][0],
                        "is_pano": sample["is_pano"]
                    })
                    processed_thumbs.add(sample["thumb_original_url"])
        print(f"Found {len(clean_data)} unique images")
        return clean_data


    def get_data(self, lat, lon, output_path, file_name=None, max_images=None, radius=10,):
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not file_name:
            json_file_name = f"mapillary_{lat}_{lon}".replace(".", "")
            image_file_name = f"mapillary_{lat}_{lon}".replace(".", "")
        else:
            json_file_name = file_name
            image_file_name = file_name

        bbox = self.__get_bbox(lat, lon, radius, show_plot=False)
        data = self.__get_image_data_by_bbox(*bbox, show_plot=False)
        data = self.__process_image_data(data)

        data_no_pano = [sample for sample in data if not sample["is_pano"]]

        json_file_name = os.path.join(output_path, json_file_name)
        image_file_name = os.path.join(output_path, image_file_name)

        if len(data) > max_images:
            print(f"Found {len(data)} images, downloading {max_images} images")
            if len(data_no_pano) > max_images:
                data = random.sample(data_no_pano, max_images)
            else:
                data = data_no_pano + random.sample([sample for sample in data if sample["is_pano"]],
                                                    max_images - len(data_no_pano))
        else:
            print(f"Found {len(data)} images, downloading all images")

        number_of_locations_downloaded = len(data)

        for idx, sample in tqdm(enumerate(data), desc=f"Downloading Images from Mapillary @ {lat},{lon}",
                                total=len(data)):
            
            try:
                self.__get_image_by_url(sample["thumb_original_url"], image_file_name + "_" + str(idx) + ".jpg")

                json_data = {
                "lat": sample["lat"],
                "lon": sample["lon"],
                "img_path": image_file_name,
                "img_url": sample["thumb_original_url"]
                }
                with open(json_file_name + "_" + str(idx) + ".json", "w") as f:
                    json.dump(json_data, f, indent=4)

            except Exception as e:
                with open(self.error_log, "a") as f:
                    f.write(f"--Error: {e} in {sample}\n")
            

        print(f"Downloaded {len(data)} images from Mapillary @ {lat},{lon}")
        print(f"Data saved to {output_path}")
        return number_of_locations_downloaded



if __name__ == "__main__":
    with open("api_key.txt", "r") as f:
        api_key = f.read().strip()


    mi = MapillaryInterface(api_key)

    lat = -17.844600
    lon = 31.089311

    print(f"Downloading images from Mapillary @ {lat},{lon}")

    mi.get_data(lat, lon, "data", max_images=100, radius=1)

