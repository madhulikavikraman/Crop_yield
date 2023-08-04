#docker build -t crop-yield-app .
#docker run -p 8000:8000 -v C <path>/data:/app/data -v <path>/model:/app/model crop-yield-app

import os
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

class InputData(BaseModel):
    District: str
    Crop_Year: int
    Crop: str
    Season: str
    Area: float
    Temperature: float
    Rainfall: float
    Soil_pH: float

datapath = './data/cropsb.csv'
modelpath = './model/model.xgb'

# Load data
data = pd.read_csv(datapath)
data = data.drop(['State_Name'], axis=1)
data = data.dropna()
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
cat_cols = ['District_Name', 'Season', 'Crop']
num_cols = ['Area', 'Production', 'Crop_Year', 'tsavg', 'rain', 'soil']
for col in cat_cols:
    data[col] = data[col].str.strip()
    data[col] = data[col].astype('category').cat.codes

X = data.drop('Production', axis=1)
y = data['Production']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
imp = SimpleImputer(strategy='mean')
X_imp = imp.fit_transform(X.iloc[:, 1:])
X_train_imp = imp.transform(X_train.iloc[:, 1:])
X_test_imp = imp.transform(X_test.iloc[:, 1:])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_imp)
X_train_scaled = scaler.transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)
# Check if model directory is empty

if os.path.exists('./model') and os.listdir('./model'):
    # Load existing model
    model = xgb.Booster(model_file=modelpath)
else:
    
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.1,
        'n_estimators': 100
    }

    model = xgb.train(params, dtrain)
    train_predictions = model.predict(dtrain)
    train_r2 = r2_score(y_train, train_predictions)

    test_predictions = model.predict(dtest)
    test_r2 = r2_score(y_test, test_predictions)

    print("Train R-squared score:", train_r2)
    print("Test R-squared score:", test_r2)

    # Save the model
    os.makedirs('./model', exist_ok=True)
    model.save_model(model_filepath)

@app.post("/predict")
def predict(d: InputData):
    district = d.District
    year = d.Crop_Year
    crop = d.Crop
    season =d.Season
    area = d.Area
    tsavg = d.Temperature
    rain = d.Rainfall
    soil = d.Soil_pH

    dist_code = pd.Series(district).astype('category').cat.codes[0]
    crop_code = pd.Series(crop).astype('category').cat.codes[0]
    season_code = pd.Series(season).astype('category').cat.codes[0]

    user_input = pd.DataFrame({
        'District_Name': [dist_code],
        'Crop_Year': [year],
        'Season': [season_code],
        'Crop': [crop_code],
        'Area': [area],
        'tsavg': [tsavg],
        'rain': [rain],
        'soil': [soil]
    })

    user_input_imp = imp.transform(user_input.iloc[:, 1:])
    user_input_scaled = scaler.transform(user_input_imp)
    dinput = xgb.DMatrix(user_input_scaled)

    prediction = model.predict(dinput)

    output_data = {
        'Predicted_Production': float(prediction[0])
    }

    return output_data
