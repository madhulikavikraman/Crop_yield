# Crop Yield Prediction API
 
A Python-based machine learning application that predicts crop production yield using an **XGBoost** regression model, served via a **FastAPI** REST API, and containerized with **Docker**.
 
---
 
## Overview
 
This project takes agricultural and environmental inputs — such as crop type, season, area, temperature, rainfall, and soil pH — and returns a predicted crop production value. The model is trained on historical crop data (`cropsb.csv`) and persisted so that subsequent container startups skip re-training.
 
---
 
## Project Structure
 
```
Crop_yield/
├── app.py                    # FastAPI application & XGBoost model logic
├── req.py                    # Helper/request script
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker container definition
├── cropsb.csv                # Crop dataset
├── model.xgb                 # Pre-trained XGBoost model
└── crop_yield_project.zip    # Full project archive
```
 
---
 
## Getting Started
 
### Prerequisites
 
- [Docker](https://www.docker.com/) installed on your machine.
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/madhulikavikraman/Crop_yield.git
cd Crop_yield
```
 
### 2. Build the Docker Image
 
```bash
docker build -t crop-yield-app .
```
 
### 3. Run the Container
 
```bash
docker run -p 8000:8000 \
  -v <your-path>/data:/app/data \
  -v <your-path>/model:/app/model \
  crop-yield-app
```
 
Replace `<your-path>` with the absolute path to your local `data` and `model` directories.
 
The API will be available at: **`http://localhost:8000`**
 
---
 
## API Usage
 
### `POST /predict`
 
Predicts crop production based on input features.
 
**Request Body (JSON):**
 
```json
{
  "District": "Thrissur",
  "Crop_Year": 2022,
  "Crop": "Rice",
  "Season": "Kharif",
  "Area": 150.0,
  "Temperature": 28.5,
  "Rainfall": 1200.0,
  "Soil_pH": 6.5
}
```
 
**Response:**
 
```json
{
  "Predicted_Production": 432.76
}
```
 
You can test the API interactively via the auto-generated Swagger UI at:
**`http://localhost:8000/docs`**
 
---
 
## Model Details
 
| Property | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| Objective | `reg:squarederror` |
| Max Depth | 3 |
| Learning Rate | 0.1 |
| Estimators | 100 |
| Evaluation Metric | R² Score |
| Preprocessing | MinMaxScaler + SimpleImputer (mean strategy) |
 
The model trains on first startup if no saved model is found in the `/app/model` volume. Otherwise, the pre-trained model is loaded directly from `model.xgb`.
 
---
 
## Tech Stack
 
- **Python 3.9**
- **FastAPI** – REST API framework
- **XGBoost** – Gradient boosting model
- **scikit-learn** – Preprocessing & evaluation
- **Pandas / NumPy** – Data handling
- **Uvicorn** – ASGI server
- **Docker** – Containerization
 
---
 
## Docker Details
 
```dockerfile
FROM python:3.9
WORKDIR /app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
 
Two Docker volumes are mounted for persistence:
- `/app/data` — holds the crop dataset (`cropsb.csv`)
- `/app/model` — stores the trained model (`model.xgb`) so it survives container restarts
 
---
 
## License
 
This project is open source. Feel free to fork and adapt it for your own use.
