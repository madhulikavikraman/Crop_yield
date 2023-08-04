import requests
import json


url = 'http://127.0.0.1:8000/predict'

district = input("Enter the district: ")
crop_year = int(input("Enter the crop year: "))
crop = input("Enter the crop: ")
season = input("Enter the season: ")
area = float(input("Enter the area: "))
temperature = float(input("Enter the temperature: "))
rainfall = float(input("Enter the rainfall: "))
soil_ph = float(input("Enter the soil pH: "))

df = {
    "District": district,
    "Crop_Year": crop_year,
    "Crop": crop,
    "Season": season,
    "Area": area,
    "Temperature": temperature,
    "Rainfall": rainfall,
    "Soil_pH": soil_ph
}

json_df = json.dumps(df)
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json_df, headers=headers)
response_data = response.json()

print(response_data)
