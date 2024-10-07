from geopy.geocoders import Nominatim
import ssl
import certifi


def get_coordinates(country):
    print(f"Getting coordinates for {country}")
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    geolocator = Nominatim(user_agent="geoapiExercises", ssl_context=ssl_context)
    location = geolocator.geocode(country)
    return location.latitude, location.longitude


