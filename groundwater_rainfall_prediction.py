import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_rainfall_data(rainfall_file='district wise rainfall normal.csv'):
    """Load and process rainfall data"""
    try:
        df_rainfall = pd.read_csv(rainfall_file)
        print("✅ Rainfall data loaded successfully")
        return df_rainfall
    except FileNotFoundError:
        print(f"⚠️  Rainfall file {rainfall_file} not found.")
        return None

def create_spatial_model(file_path='CGWB_data_main_cleaned.csv'):
    """Create spatial model using KNN based on lat/lon"""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None, None

    id_vars = ['STATE', 'DISTRICT', 'LAT', 'LON', 'SITE_TYPE', 'WLCODE']
    df_long = pd.melt(df, id_vars=id_vars, var_name='Date', value_name='Water_Level')

    df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')
    df_long.dropna(subset=['Water_Level', 'LAT', 'LON', 'Date'], inplace=True)

    df_recent = df_long.loc[df_long.groupby('WLCODE')['Date'].idxmax()]

    X = df_recent[['LAT', 'LON']]
    y = df_recent['Water_Level']

    knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn_model.fit(X, y)

    print("✅ Spatial model trained (KNN)")
    return knn_model, df_long

def get_rainfall_for_district(district, rainfall_df):
    """Get rainfall data for a specific district"""
    if rainfall_df is None:
        return None
    
    district_data = rainfall_df[rainfall_df['DISTRICT'].str.upper() == district.upper()]
    if not district_data.empty:
        return district_data.iloc[0]
    return None

def prepare_lstm_data_with_rainfall(yearly_avg, rainfall_series, seq_len=3):
    """Prepare LSTM data combining water level and rainfall features"""
    water_level = yearly_avg['Water_Level'].values
    
    if rainfall_series is not None:
        rainfall = rainfall_series.values if hasattr(rainfall_series, 'values') else rainfall_series
        # Normalize both to same length
        min_len = min(len(water_level), len(rainfall))
        water_level = water_level[:min_len]
        rainfall = rainfall[:min_len]
        # Stack features
        combined = np.column_stack([water_level, rainfall])
        n_features = 2
    else:
        combined = water_level.reshape(-1, 1)
        n_features = 1

    X, y = [], []
    for i in range(len(combined) - seq_len):
        X.append(combined[i:i + seq_len])
        y.append(water_level[i + seq_len])

    return np.array(X), np.array(y), n_features

def build_cnn_lstm_multifeature(seq_len, n_features=1):
    """Build CNN+LSTM model for multifeature input"""
    model = Sequential([
        Conv1D(32, kernel_size=2, activation='relu', input_shape=(seq_len, n_features)),
        MaxPooling1D(pool_size=1),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def cnn_lstm_predict_with_rainfall(yearly_avg, rainfall_data, future_year, seq_len=3, output_dir="outputs"):
    """Train CNN+LSTM model with rainfall features for predictions"""
    
    X, y, n_features = prepare_lstm_data_with_rainfall(yearly_avg, rainfall_data, seq_len)
    
    if len(X) == 0:
        print("⚠️  Not enough data for training")
        return None, None, None
    
    scaler_water = MinMaxScaler()
    scaler_rainfall = MinMaxScaler() if n_features > 1 else None
    
    X_scaled = X.copy().astype(float)
    
    # Scale each feature separately
    X_scaled[:, :, 0] = scaler_water.fit_transform(X[:, :, 0].reshape(-1, 1)).reshape(X[:, :, 0].shape)
    
    if n_features > 1:
        X_scaled[:, :, 1] = scaler_rainfall.fit_transform(X[:, :, 1].reshape(-1, 1)).reshape(X[:, :, 1].shape)

    model = build_cnn_lstm_multifeature(seq_len, n_features)

    # Train model
    history = model.fit(X_scaled, y, epochs=100, batch_size=4, verbose=0)

    # ----- ACCURACY METRICS -----
    y_pred_train = model.predict(X_scaled, verbose=0)
    mse = mean_squared_error(y, y_pred_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)

    print("\n📊 TRAINING ACCURACY (With Rainfall Features)")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    # ----- TRAINING LOSS PLOT -----
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'])
    plt.title("CNN + LSTM Model Loss Curve (With Rainfall)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    loss_plot_path = os.path.join(output_dir, "training_loss_with_rainfall.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ----- FUTURE PREDICTIONS -----
    future_predictions = {}
    last_seq = X[-1]
    last_year = yearly_avg["Year"].max()

    for year in range(last_year + 1, future_year + 1):
        seq_scaled = last_seq.copy().astype(float)
        seq_scaled[:, 0] = scaler_water.transform(last_seq[:, 0].reshape(-1, 1)).flatten()
        if n_features > 1:
            seq_scaled[:, 1] = scaler_rainfall.transform(last_seq[:, 1].reshape(-1, 1)).flatten()
        
        seq_scaled = seq_scaled.reshape(1, seq_len, n_features)
        next_pred = model.predict(seq_scaled, verbose=0)[0][0]

        future_predictions[year] = next_pred
        
        # Update sequence with prediction and estimated rainfall
        if n_features > 1:
            estimated_rainfall = np.mean(last_seq[1:, 1])
            last_seq = np.vstack([last_seq[1:], [next_pred, estimated_rainfall]])
        else:
            last_seq = np.vstack([last_seq[1:], [[next_pred]]])

    return future_predictions, loss_plot_path, (mse, rmse, mae, r2)

def predict_until_year_with_rainfall(latitude, longitude, district, future_year, model, df_long, rainfall_df, output_dir="outputs"):
    """Main prediction function integrating rainfall data"""
    if model is None or df_long is None:
        return "Model unavailable."

    df = df_long

    input_data = pd.DataFrame([[latitude, longitude]], columns=['LAT', 'LON'])
    distances, indices = model.kneighbors(input_data, n_neighbors=1)

    nearest_lat, nearest_lon = model._fit_X[indices[0][0]]
    nearest_well = df[(df['LAT'] == nearest_lat) & (df['LON'] == nearest_lon)]['WLCODE'].iloc[0]

    ts = df[df['WLCODE'] == nearest_well].copy()
    ts["Year"] = ts["Date"].dt.year
    yearly_avg = ts.groupby("Year")["Water_Level"].mean().reset_index()

    # Get rainfall data for the district
    rainfall_yearly = None
    if district and rainfall_df is not None:
        rainfall_data = get_rainfall_for_district(district, rainfall_df)
        if rainfall_data is not None:
            # Create yearly rainfall series (using annual rainfall)
            rainfall_yearly = pd.Series([rainfall_data['ANNUAL']] * len(yearly_avg), index=yearly_avg.index)
            print(f"✅ Using annual rainfall data for district: {district}")

    # Predictions with rainfall + accuracy + training loss image
    predictions, loss_img, scores = cnn_lstm_predict_with_rainfall(yearly_avg, rainfall_yearly, future_year)

    if predictions is None:
        return nearest_well, {}, "", "", scores

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_avg["Year"], yearly_avg["Water_Level"], marker="o", label="Historical", linewidth=2)
    plt.plot(list(predictions.keys()), list(predictions.values()), marker="o",
             linestyle="--", color="red", label="Predicted (with Rainfall)", linewidth=2)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Groundwater Level (m)", fontsize=12)
    plt.title(f"Groundwater Prediction (CNN+LSTM+Rainfall) for Well {nearest_well}\nDistrict: {district}", fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)

    plot_path = os.path.join(output_dir, f"prediction_rainfall_{nearest_well}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return nearest_well, predictions, plot_path, loss_img, scores

# Main execution
if __name__ == "__main__":
    print("🌍 Groundwater Level Prediction with Rainfall Integration\n")
    print("=" * 60)
    
    # Load models and data
    spatial_model, df_long = create_spatial_model('CGWB_data_main_cleaned.csv')
    rainfall_df = load_rainfall_data('district wise rainfall normal.csv')

    if spatial_model:
        print("\n📍 Enter location coordinates and district:")
        lat = float(input("Enter latitude: "))
        lon = float(input("Enter longitude: "))
        district = input("Enter district name: ").strip()
        year = int(input("Predict groundwater level up to year: "))

        well, preds, plot_file, loss_file, acc = predict_until_year_with_rainfall(
            lat, lon, district, year, spatial_model, df_long, rainfall_df
        )

        print("\n--- 📌 Groundwater Future Prediction (with Rainfall) ---")
        print(f"Nearest Well: {well}")
        print(f"District: {district}\n")

        for yr, lvl in preds.items():
            print(f"{yr} → {lvl:.2f} meters")

        print("\n--- 📊 Accuracy Scores ---")
        print(f"MSE  : {acc[0]:.3f}")
        print(f"RMSE : {acc[1]:.3f}")
        print(f"MAE  : {acc[2]:.3f}")
        print(f"R²   : {acc[3]:.3f}")

        print(f"\n📁 Prediction Image Saved: {plot_file}")
        print(f"📁 Training Loss Image Saved: {loss_file}")
        print("\n" + "=" * 60)
    else:
        print("❌ Failed to initialize model")
