import requests
import json

from PIL import Image
import numpy as np
import os

def get_panoId(lat, lon):
    with open('./GoogleStreetMapDataCollection/.googleApiKey','r') as f:
        api_key = f.read()

    url = F"https://maps.googleapis.com/maps/api/streetview/metadata?location={round(lat,3)}%2C{round(lon,3)}&key={api_key}"

    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    
    json_data = json.loads(response.text)

    if 'pano_id' not in json_data:
        print("Error: Pano ID not found")
        print(response.text)
        raise Exception("Pano ID not found")

    return json_data['pano_id']


def get_panoImage(panoId,x,y,zoom,save_path = "./Test.jpeg"):


    url = F"https://streetviewpixels-pa.googleapis.com/v1/tile?cb_client=maps_sv.tactile&panoid={panoId}&x={x}&y={y}&zoom={zoom}&nbt=1&fover=2"
    payload = {}
    headers = {}

    response = requests.request("GET", url, headers=headers, data=payload)
    # Save the image/jpeg

    if response.headers['Content-Type'] == 'image/jpeg':
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        print("Error: Image not found")
        print(panoId)
        print(response.headers)
        print(response.text)
        raise Exception("Image not found")
    
    return save_path

def get_full_pano(lat,lon):
    panoId = get_panoId(lat, lon)
    tiles = []
    for x in range(0,4):
        for y in range(0,4):
            tiles.append(get_panoImage(panoId,x,y,3,f"./GoogleStreetMapDataCollection/temp/temp_{x}_{y}.jpeg"))

    # Combine the tiles into a single image
    

    images = [Image.open(i) for i in tiles]
    widths, heights = zip(*(i.size for i in images))

    total_width = images[0].size[0] * 4
    max_height = images[0].size[1] * 4

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    y_offset = 0
    for y in range(0,4):
        for x in range(0,4):
            new_im.paste(images[x*4 + y], (x_offset,y_offset))
            x_offset += images[x*4 + y].size[0]
        x_offset = 0
        y_offset += images[x*4].size[1]
        

    new_im.save('full_pano.jpeg')

    



if __name__ == '__main__':
    get_full_pano(36.665773445152055, 139.7578901052475)
