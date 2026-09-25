import requests
import certifi

url = "https://cds-beta.climate.copernicus.eu/api/catalogue/v1/collections"
print(f"Testing HTTPS connection to {url}")
print(f"Using certifi store: {certifi.where()}")
try:
    response = requests.get(url, timeout=10)
    print(f"Success! HTTP Status: {response.status_code}")
except Exception as e:
    print(f"Failed: {e}")
