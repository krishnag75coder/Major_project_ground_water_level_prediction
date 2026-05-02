"""
Groundwater Prediction API (Final Stable Version)

Run:
    uvicorn app:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional

app = FastAPI(title="Groundwater Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data
DATA = {"df_long": None}


# ------------------ REQUEST MODEL ------------------ #
class PredictRequest(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    district: Optional[str] = None
    target_year: Optional[int] = None


# ------------------ LOAD DATA ------------------ #
@app.on_event("startup")
def load_data():
    try:
        df = pd.read_csv("CGWB_data_main_cleaned.csv")
        print("✅ Dataset loaded:", df.shape)
    except Exception as e:
        print("❌ Dataset loading failed:", e)
        return

    # Required columns check
    if "LAT" not in df.columns or "LON" not in df.columns:
        raise Exception("Missing LAT/LON columns")

    if "WLCODE" not in df.columns:
        df["WLCODE"] = df.index.astype(str)

    id_vars = [c for c in ["STATE", "DISTRICT", "LAT", "LON", "WLCODE"] if c in df.columns]

    # Convert wide → long
    df_long = pd.melt(df, id_vars=id_vars, var_name="Date", value_name="Water_Level")

    # Convert types safely
    df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce", dayfirst=True)
    df_long["Water_Level"] = pd.to_numeric(df_long["Water_Level"], errors="coerce")

    # Drop invalid rows
    df_long.dropna(subset=["Water_Level", "LAT", "LON", "Date"], inplace=True)

    if df_long.empty:
        raise Exception("Dataset empty after cleaning")

    DATA["df_long"] = df_long
    print("✅ Data prepared successfully")


# ------------------ HOME ------------------ #
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Groundwater Prediction API</h2>
    <p>POST /predict with:</p>
    <pre>{
      "district": "Ghaziabad",
      "target_year": 2035
    }</pre>
    """


# ------------------ CORE PREDICTION ------------------ #
def generate_prediction(years, values, target_year):
    # Convert to numpy
    years = np.array(years)
    values = np.array(values)

    # Force numeric
    try:
        years = years.astype(float)
        values = values.astype(float)
    except:
        raise HTTPException(status_code=500, detail="Non-numeric data found")

    # Remove NaN
    mask = ~np.isnan(years) & ~np.isnan(values)
    years = years[mask]
    values = values[mask]

    if len(years) == 0:
        raise HTTPException(status_code=500, detail="No valid data")

    last_year = int(years[-1])

    # If no target year → latest
    if target_year is None:
        return {
            "predicted_year": last_year,
            "predicted_value": float(values[-1]),
            "method": "latest"
        }

    if target_year <= last_year:
        raise HTTPException(status_code=400, detail="Target year must be future")

    # Linear trend prediction
    if len(years) >= 2:
        slope, intercept = np.polyfit(years, values, 1)

        future_years = list(range(last_year + 1, target_year + 1))
        future_preds = [float(slope * y + intercept) for y in future_years]

        return {
            "base_last_year": last_year,
            "target_year": target_year,
            "method": "linear_trend",
            "predictions": [
                {"year": y, "predicted_value": v}
                for y, v in zip(future_years, future_preds)
            ]
        }

    # Fallback
    return {
        "predicted_year": last_year,
        "predicted_value": float(values[-1]),
        "method": "fallback"
    }


# ------------------ LOCATION PREDICTION ------------------ #
def predict_by_location(lat, lon, target_year):
    df = DATA["df_long"]

    recent = df.loc[df.groupby("WLCODE")["Date"].idxmax()]

    distances = np.sqrt((recent["LAT"] - lat)**2 + (recent["LON"] - lon)**2)
    idx = distances.idxmin()

    well = recent.loc[idx]
    well_id = well["WLCODE"]
    district = well.get("DISTRICT", None)

    well_data = df[df["WLCODE"] == well_id].copy()
    well_data["Year"] = well_data["Date"].dt.year

    yearly = well_data.groupby("Year")["Water_Level"].mean().dropna()

    result = generate_prediction(yearly.index, yearly.values, target_year)

    return {
        "type": "location",
        "well_id": str(well_id),
        "district": district,
        **result
    }


# ------------------ DISTRICT PREDICTION ------------------ #
def predict_by_district(district_name, target_year):
    df = DATA["df_long"]

    df_district = df[df["DISTRICT"].str.lower() == district_name.lower()]

    if df_district.empty:
        raise HTTPException(status_code=404, detail="District not found")

    df_district["Year"] = df_district["Date"].dt.year

    yearly = df_district.groupby("Year")["Water_Level"].mean().dropna()

    result = generate_prediction(yearly.index, yearly.values, target_year)

    return {
        "type": "district",
        "district": district_name,
        **result
    }


# ------------------ API ------------------ #
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if req.district:
            return JSONResponse(content=predict_by_district(req.district, req.target_year))

        if req.latitude is not None and req.longitude is not None:
            return JSONResponse(content=predict_by_location(req.latitude, req.longitude, req.target_year))

        raise HTTPException(status_code=400, detail="Provide district OR latitude/longitude")

    except HTTPException:
        raise
    except Exception as e:
        print("❌ ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ------------------ HEALTH ------------------ #
@app.get("/health")
def health():
    return {"status": "ok" if DATA["df_long"] is not None else "loading"}