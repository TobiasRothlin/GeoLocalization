from datetime import datetime
import os


def createMDFileRaport(data):
    text = "# Meta Data File Raport\n\n"
    text += "## Test Data\n"
    text += f"### JSON\n"
    text += f"    - Total: {data['Test']['JSON']}\n"
    text += f"    - Meta Data: {data['Test']['JSON_META']}\n"
    text += f"### JPG\n"
    text += f"    - Total: {data['Test']['JPG']}\n"
    text += f"    - Meta Data: {data['Test']['JPG_META']}\n"
    text += f"### JPEG\n"
    text += f"    - Total: {data['Test']['JPEG']}\n"
    text += f"    - Meta Data: {data['Test']['JPEG_META']}\n"

    text += f"### Meta Data Removed\n"
    for f in data['Test']['META_REMOVED']:
        text += f"    - {f}\n"



    text += f"### Validation\n"
    text += f"    - City: {data['Test']['Checks']['city']} | {data['Test']['Total']} ({100*data['Test']['Checks']['city']/data['Test']['Total']}%)\n"
    text += f"    - Country: {data['Test']['Checks']['country']} | {data['Test']['Total']} ({100*data['Test']['Checks']['country']/data['Test']['Total']}%)\n"
    text += f"    - Latitude: {data['Test']['Checks']['lat']} | {data['Test']['Total']} ({100*data['Test']['Checks']['lat']/data['Test']['Total']}%)\n"
    text += f"    - Longitude: {data['Test']['Checks']['lon']} | {data['Test']['Total']} ({100*data['Test']['Checks']['lon']/data['Test']['Total']}%)\n"
    text += f"    - Predicted Population Area: {data['Test']['Checks']['PredictedPopulationArea']} | {data['Test']['Total']} ({100*data['Test']['Checks']['PredictedPopulationArea']/data['Test']['Total']}%)\n"
    text += f"    - Predicted Region: {data['Test']['Checks']['PredictedRegion']} | {data['Test']['Total']} ({100*data['Test']['Checks']['PredictedRegion']/data['Test']['Total']}%)\n"
    text += f"    - Did Reverse Geo Location: {data['Test']['Checks']['DidReverseGeoLocation']} | {data['Test']['Total']} ({100*data['Test']['Checks']['DidReverseGeoLocation']/data['Test']['Total']}%)\n"
    text += f"    - Did Classification: {data['Test']['Checks']['DidClassification']} | {data['Test']['Total']} ({100*data['Test']['Checks']['DidClassification']/data['Test']['Total']}%)\n"
    text += f"    - Image Exists: {data['Test']['Checks']['ImageExists']} | {data['Test']['Total']} ({100*data['Test']['Checks']['ImageExists']/data['Test']['Total']}%)\n"
    text += f"    - Can Open Image: {data['Test']['Checks']['CanOpenImage']} | {data['Test']['Total']} ({100*data['Test']['Checks']['CanOpenImage']/data['Test']['Total']}%)\n"

    text += "## Train Data\n"
    text += f"### JSON\n"
    text += f"    - Total: {data['Train']['JSON']}\n"
    text += f"    - Meta Data: {data['Train']['JSON_META']}\n"
    text += f"### JPG\n"
    text += f"    - Total: {data['Train']['JPG']}\n"
    text += f"    - Meta Data: {data['Train']['JPG_META']}\n"
    text += f"### JPEG\n"
    text += f"    - Total: {data['Train']['JPEG']}\n"
    text += f"    - Meta Data: {data['Train']['JPEG_META']}\n"

    text += f"### Meta Data Removed\n"
    for f in data['Train']['META_REMOVED']:
        text += f"    - {f}\n"


    text += f"### Validation\n"
    text += f"    - City: {data['Train']['Checks']['city']} | {data['Train']['Total']} ({100*data['Train']['Checks']['city']/data['Train']['Total']}%)\n"
    text += f"    - Country: {data['Train']['Checks']['country']} | {data['Train']['Total']} ({100*data['Train']['Checks']['country']/data['Train']['Total']}%)\n"
    text += f"    - Latitude: {data['Train']['Checks']['lat']} | {data['Train']['Total']} ({100*data['Train']['Checks']['lat']/data['Train']['Total']}%)\n"
    text += f"    - Longitude: {data['Train']['Checks']['lon']} | {data['Train']['Total']} ({100*data['Train']['Checks']['lon']/data['Train']['Total']}%)\n"
    text += f"    - Predicted Population Area: {data['Train']['Checks']['PredictedPopulationArea']} | {data['Train']['Total']} ({100*data['Train']['Checks']['PredictedPopulationArea']/data['Train']['Total']}%)\n"
    text += f"    - Predicted Region: {data['Train']['Checks']['PredictedRegion']} | {data['Train']['Total']} ({100*data['Train']['Checks']['PredictedRegion']/data['Train']['Total']}%)\n"
    text += f"    - Did Reverse Geo Location: {data['Train']['Checks']['DidReverseGeoLocation']} | {data['Train']['Total']} ({100*data['Train']['Checks']['DidReverseGeoLocation']/data['Train']['Total']}%)\n"
    text += f"    - Did Classification: {data['Train']['Checks']['DidClassification']} | {data['Train']['Total']} ({100*data['Train']['Checks']['DidClassification']/data['Train']['Total']}%)\n"
    text += f"    - Image Exists: {data['Train']['Checks']['ImageExists']} | {data['Train']['Total']} ({100*data['Train']['Checks']['ImageExists']/data['Train']['Total']}%)\n"
    text += f"    - Can Open Image: {data['Train']['Checks']['CanOpenImage']} | {data['Train']['Total']} ({100*data['Train']['Checks']['CanOpenImage']/data['Train']['Total']}%)\n"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = "/home/tobias.rothlin/GeoLocalization/src/DGX1/Reports"
    file_path = os.path.join(base_path, f"Meta_Data_Raport_{timestamp}.md")
    with open(file_path, "w") as f:
        f.write(text)