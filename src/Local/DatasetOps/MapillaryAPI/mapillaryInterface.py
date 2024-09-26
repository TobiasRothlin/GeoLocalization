import requests
import mercantile
import json

from tqdm import tqdm

class MapillaryInterface:

    def __init__(self, api_key):
        self.access_token = api_key



    def get_bbox(self, lat, lon, radius_m=100):
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError("Latitude and Longitude are out of range.")

        # Convert radius to degrees
        radius_deg = radius_m / 111139

        west = lon - radius_deg
        south = lat - radius_deg
        east = lon + radius_deg
        north = lat + radius_deg
        return west, south, east, north

       

    def get_image_data_by_bbox(self, west: float, south: float, east: float, north: float):
        tiles = list(mercantile.tiles(west, south, east, north, 18))
        bbox_list = [mercantile.bounds(tile.x, tile.y, tile.z) for tile in tiles]

        image_data = []
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

            while url:
                response = requests.get(url)
                data = response.json()
                if data:
                    image_data+=data["data"]

                    url = data.get('paging', {}).get('next', None)

        return {'data': image_data}
    
    def get_image_by_url(self, url, save_path):
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)

    def process_image_data(self, image_data):
        clean_data = []
        processed_thumbs = []
        for sample in tqdm(image_data["data"], desc="Processing Image Data"):
            if sample["thumb_original_url"] not in processed_thumbs:
                clean_data.append({
                    "thumb_original_url": sample["thumb_original_url"],
                    "lat": sample["computed_geometry"]["coordinates"][1],
                    "lon": sample["computed_geometry"]["coordinates"][0],
                    "is_pano": sample["is_pano"]
                })
                processed_thumbs.append(sample["thumb_original_url"])

        return clean_data



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