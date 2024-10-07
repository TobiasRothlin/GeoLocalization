import os

import requests
import mercantile
import json

from tqdm import tqdm

import random

import numpy as np

class MapillaryInterface:

    def __init__(self, api_key):
        self.access_token = api_key

    def get_bbox(self, center_lat, center_lon, radius_km=100):
        if not (-180 <= center_lon <= 180 and -90 <= center_lat <= 90):
            raise ValueError("Latitude and Longitude are out of range.")

        R = 6371  # Earth radius in km

        lat1 = center_lat - np.degrees(radius_km / R)
        lat2 = center_lat + np.degrees(radius_km / R)
        lon1 = center_lon - np.degrees(radius_km / R / np.cos(np.radians(center_lat)))
        lon2 = center_lon + np.degrees(radius_km / R / np.cos(np.radians(center_lat)))

        return lat1, lon1, lat2, lon2

       

    def get_image_data_by_bbox(self, west: float, south: float, east: float, north: float, max_images=100):
        tiles = list(mercantile.tiles(west, south, east, north, 4))
        bbox_list = [mercantile.bounds(tile.x, tile.y, tile.z) for tile in tiles]

        image_data = []

        random.shuffle(bbox_list)
        

        for bbox in bbox_list:
            bbox_str = str(f'{bbox.west},{bbox.south},{bbox.east},{bbox.north}')


            if not (-180 <= bbox.west <= 180 and -180 <= bbox.east <= 180 and -90 <= bbox.south <= 90 and -90 <= bbox.north <= 90):
                raise ValueError("Bounding box coordinates are out of range.")
            
            fields = [
                "thumb_original_url",
                "computed_geometry",
                "geometry",
                "is_pano"
            ]
            # Rate limit for images? is 10'000 per minute
            url = f'https://graph.mapillary.com/images?access_token={self.access_token}&bbox={bbox_str}&fields={",".join(fields)}'
            
            progress_bar = tqdm( desc="Downloading Image Data", unit=" chunks")
            while url:
                response = requests.get(url)
                data = response.json()
                if data:
                    try:
                        image_data+=data["data"]
                    except KeyError as e:
                        with open("error.txt", "w") as f:
                            f.write(f"No data in response: {data.keys()}")
                            f.write(f"Response: {data}")
                        raise e
                    url = data.get('paging', {}).get('next', None)
                    progress_bar.update(1)

        progress_bar.close()
        return image_data
    
    def get_image_by_url(self, url, save_path):
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)

    def process_image_data(self, image_data):
        clean_data = []
        processed_thumbs = []
        for sample in tqdm(image_data, desc="Processing Image Data"):
            try:
                if sample["thumb_original_url"] not in processed_thumbs:
                    clean_data.append({
                        "thumb_original_url": sample["thumb_original_url"],
                        "lat": sample["geometry"]["coordinates"][1],
                        "lon": sample["geometry"]["coordinates"][0],
                        "is_pano": sample["is_pano"]
                    })
                    processed_thumbs.append(sample["thumb_original_url"])
            except KeyError as e:
                with open("error.txt", "w") as f:
                    f.write(f"KeyError: {sample.keys()}")
                    f.write(f"Sample: {sample}")
                raise e
        print(f"Found {len(clean_data)} unique images")
        return clean_data
    
    def get_data(self, lat, lon,output_path, file_name = None, radius_m=100, max_images=100):
        bbox = self.get_bbox(lat, lon, radius_m)
        data = self.get_image_data_by_bbox(*bbox, 10*max_images)
        data = self.process_image_data(data)

        data_no_pano = [sample for sample in data if not sample["is_pano"]]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if not file_name:
            json_file_name = f"mapillary_{lat}_{lon}".replace(".", "")
            image_file_name = f"mapillary_{lat}_{lon}".replace(".", "")

        
        json_file_name = os.path.join(output_path, json_file_name)
        image_file_name = os.path.join(output_path, image_file_name)

        if len(data) > max_images:
            if len(data_no_pano) > max_images:
                data = random.sample(data_no_pano, max_images)
            else:
                data = data_no_pano + random.sample([sample for sample in data if sample["is_pano"]], max_images-len(data_no_pano))
            

        number_of_locations_downloaded = len(data)

        for idx,sample in tqdm(enumerate(data), desc=f"Downloading Images from Mapillary @ {lat},{lon}", total=len(data)):
            self.get_image_by_url(sample["thumb_original_url"], image_file_name+"_"+str(idx)+".jpg")

            json_data = {
                "lat": lat,
                "lon": lon,
                "img_path": image_file_name,
            }
            with open(json_file_name+"_"+str(idx)+".json", "w") as f:
                json.dump(json_data, f, indent=4)

        print(f"Downloaded {len(data)} images from Mapillary @ {lat},{lon}")
        print(f"Data saved to {output_path}")

        return number_of_locations_downloaded


if __name__ == "__main__":
    with open(".apiTokenMapillary", "r") as f:
        api_key = f.read()


    mi = MapillaryInterface(api_key)

    bbox = mi.get_bbox(47.223627, 8.818817, radius_m=1)
    print(bbox)
    data = mi.get_image_data_by_bbox(*bbox)

    data = mi.process_image_data(data)

    mi.get_image_by_url(data[0]["thumb_original_url"], "test.jpg")

    with open("test.json", "w") as f:
        json.dump(data, f, indent=4)

        