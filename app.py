"""
Groundwater Prediction API (FINAL STABLE VERSION)

Run locally:
    uvicorn app:app --reload

Deploy (Render):
    gunicorn -k uvicorn.workers.UvicornWorker app:app
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional

app = FastAPI(title="Groundwater Prediction API")

# Enable CORS (frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage
DATA = {
    "df_long": None
}

# Request schema
class PredictRequest(BaseModel):
    latitude: float
    longitude: float
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

    # Ensure required columns
    if "LAT" not in df.columns or "LON" not in df.columns:
        raise Exception("Dataset must contain LAT and LON columns")

    # Create fallback well ID
    if "WLCODE" not in df.columns:
        df["WLCODE"] = df.index.astype(str)

    id_vars = [c for c in ["STATE", "DISTRICT", "LAT", "LON", "WLCODE"] if c in df.columns]

    # Convert to long format
    df_long = pd.melt(df, id_vars=id_vars, var_name="Date", value_name="Water_Level")

    # Convert date safely
    df_long["Date"] = pd.to_datetime(df_long["Date"], errors="coerce", dayfirst=True)

    # Convert water level to numeric (CRITICAL FIX)
    df_long["Water_Level"] = pd.to_numeric(df_long["Water_Level"], errors="coerce")

    # Drop invalid rows
    df_long.dropna(subset=["Water_Level", "LAT", "LON", "Date"], inplace=True)

    if df_long.empty:
        raise Exception("Dataset empty after cleaning")

    DATA["df_long"] = df_long
    print("✅ Data preprocessing complete")


# ------------------ HOME ------------------ #
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Groundwater Prediction API</h2>
    <p>Use POST /predict</p>
    <pre>{
      "latitude": 28.53,
      "longitude": 77.39,
      "district": "Ghaziabad",
      "target_year": 2040
    }</pre>
    """


# ------------------ PREDICTION LOGIC ------------------ #
def predict_logic(lat, lon, district=None, target_year=None):
    df_long = DATA.get("df_long")

    if df_long is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    # Filter by district if provided
    if district and "DISTRICT" in df_long.columns:
        df_filtered = df_long[df_long["DISTRICT"].str.lower() == district.lower()]
        if not df_filtered.empty:
            df_long = df_filtered

    # Latest records per well
    recent = df_long.loc[df_long.groupby("WLCODE")["Date"].idxmax()]

    if recent.empty:
        raise HTTPException(status_code=404, detail="No wells found")

    # Find nearest well
    distances = np.sqrt((recent["LAT"] - lat)**2 + (recent["LON"] - lon)**2)
    idx = distances.idxmin()

    well = recent.loc[idx]
    well_id = well["WLCODE"]
    district_name = well.get("DISTRICT", "Unknown")
    state_name = well.get("STATE", "Unknown")

    # Historical data
    well_data = df_long[df_long["WLCODE"] == well_id].copy()
    well_data["Year"] = well_data["Date"].dt.year

    yearly = well_data.groupby("Year")["Water_Level"].mean().dropna()

    if yearly.empty:
        raise HTTPException(status_code=404, detail="No historical data")

    # 🔥 CRITICAL FIX (numeric conversion)
    years = yearly.index.values.astype(float)
    values = yearly.values.astype(float)

    # Latest value if no target
    if target_year is None:
        return {
            "well_id": str(well_id),
            "district": str(district_name),
            "state": str(state_name),
            "predicted_year": int(years[-1]),
            "predicted_value": float(values[-1]),
            "method": "latest"
        }

    # Linear prediction
    if len(years) >= 2:
        try:
            slope, intercept = np.polyfit(years, values, 1)
            pred = slope * target_year + intercept
        except Exception:
            pred = values[-1]

        return {
            "well_id": str(well_id),
            "district": str(district_name),
            "state": str(state_name),
            "predicted_year": int(target_year),
            "predicted_value": float(pred),
            "method": "linear_trend"
        }

    # Fallback
    return {
        "well_id": str(well_id),
        "district": str(district_name),
        "state": str(state_name),
        "predicted_year": int(years[-1]),
        "predicted_value": float(values[-1]),
        "method": "fallback"
    }


# ------------------ API ------------------ #
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = predict_logic(
            req.latitude,
            req.longitude,
            req.district,
            req.target_year
        )
        return JSONResponse(content=result)

    except HTTPException:
        raise

    except Exception as e:
        print("❌ ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ------------------ HEALTH ------------------ #
@app.get("/health")
def health():
    return {"status": "ok" if DATA["df_long"] is not None else "loading"}