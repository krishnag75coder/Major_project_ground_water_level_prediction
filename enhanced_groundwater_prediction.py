import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

sns.set_style("whitegrid")

def load_rainfall_data(rainfall_file='district wise rainfall normal.csv'):
    """Load and process rainfall data"""
    try:
        df_rainfall = pd.read_csv(rainfall_file)
        print("[+] Rainfall data loaded successfully")
        return df_rainfall
    except FileNotFoundError:
        print(f"Warning: Rainfall file {rainfall_file} not found.")
        return None

def create_spatial_model(file_path='CGWB_data_main_cleaned.csv'):
    """Create spatial model using KNN"""
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

    print("[+] Spatial model trained (KNN)")
    return knn_model, df_long

def get_rainfall_for_district(district, rainfall_df):
    """Get rainfall for specific district"""
    if rainfall_df is None:
        return None
    
    district_data = rainfall_df[rainfall_df['DISTRICT'].str.upper() == district.upper()]
    if not district_data.empty:
        return district_data.iloc[0]
    return None

def prepare_lstm_data_with_rainfall(yearly_avg, rainfall_series, seq_len=3):
    """Prepare LSTM data with dual features"""
    water_level = yearly_avg['Water_Level'].values
    
    if rainfall_series is not None:
        rainfall = rainfall_series.values if hasattr(rainfall_series, 'values') else rainfall_series
        min_len = min(len(water_level), len(rainfall))
        water_level = water_level[:min_len]
        rainfall = rainfall[:min_len]
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
    """Build CNN+LSTM model"""
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

def plot_actual_vs_predicted(yearly_avg, y_pred_train, output_dir="outputs"):
    """Plot actual vs predicted"""
    plt.figure(figsize=(10, 6))
    years = yearly_avg['Year'].values
    actual = yearly_avg['Water_Level'].values
    
    plt.plot(years, actual, marker='o', label='Actual', linewidth=2, markersize=8, color='blue')
    plt.plot(years[:len(y_pred_train)], y_pred_train.flatten(), marker='s', label='Predicted (Training)', 
             linewidth=2, markersize=7, color='orange', linestyle='--')
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Water Level (meters)', fontsize=12, fontweight='bold')
    plt.title('Actual vs Predicted Groundwater Levels', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "01_actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {plot_path}")
    return plot_path

def plot_scatter_actual_vs_predicted(yearly_avg, y_pred_train, output_dir="outputs"):
    """Scatter plot of actual vs predicted"""
    actual = yearly_avg['Water_Level'].values[:len(y_pred_train)]
    predicted = y_pred_train.flatten()
    
    plt.figure(figsize=(9, 7))
    plt.scatter(actual, predicted, alpha=0.6, s=100, color='green', edgecolors='darkgreen')
    
    # Add perfect prediction line
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Water Level (meters)', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Water Level (meters)', fontsize=12, fontweight='bold')
    plt.title('Scatter Plot: Actual vs Predicted', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "02_scatter_actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {plot_path}")
    return plot_path

def plot_residuals_boxplot(yearly_avg, y_pred_train, output_dir="outputs"):
    """Residual box plot"""
    actual = yearly_avg['Water_Level'].values[:len(y_pred_train)]
    predicted = y_pred_train.flatten()
    residuals = actual - predicted
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot
    ax1.boxplot([residuals], labels=['Residuals'])
    ax1.set_ylabel('Residual (meters)', fontsize=11, fontweight='bold')
    ax1.set_title('Residual Box Plot', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Distribution plot
    ax2.hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Residual (meters)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "03_residuals_boxplot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {plot_path}")
    return plot_path

def plot_future_trend(yearly_avg, predictions, output_dir="outputs"):
    """Future groundwater level prediction trend"""
    historical_years = yearly_avg['Year'].values
    historical_levels = yearly_avg['Water_Level'].values
    
    future_years = list(predictions.keys())
    future_levels = list(predictions.values())
    
    plt.figure(figsize=(12, 7))
    
    # Historical
    plt.plot(historical_years, historical_levels, marker='o', label='Historical Data', 
             linewidth=2.5, markersize=8, color='blue')
    
    # Future
    plt.plot(future_years, future_levels, marker='s', label='Future Prediction', 
             linewidth=2.5, markersize=8, color='red', linestyle='--')
    
    # Highlight transition
    plt.plot([historical_years[-1], future_years[0]], [historical_levels[-1], future_levels[0]], 
             'k--', linewidth=1.5, alpha=0.5)
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Groundwater Level (meters)', fontsize=12, fontweight='bold')
    plt.title('Future Groundwater Level Prediction Trend', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=historical_years[-1], color='gray', linestyle=':', alpha=0.7, label='Transition')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "04_future_trend.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {plot_path}")
    return plot_path

def save_predictions_data(predictions, yearly_avg, output_dir="outputs"):
    """Export predictions to CSV"""
    df_predictions = pd.DataFrame({
        'Year': list(predictions.keys()),
        'Predicted_Water_Level_m': list(predictions.values()),
        'Type': ['Future'] * len(predictions)
    })
    
    df_historical = pd.DataFrame({
        'Year': yearly_avg['Year'].values,
        'Predicted_Water_Level_m': yearly_avg['Water_Level'].values,
        'Type': ['Historical'] * len(yearly_avg)
    })
    
    # Combined
    df_combined = pd.concat([df_historical, df_predictions], ignore_index=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save all formats
    csv_path = os.path.join(output_dir, "groundwater_predictions.csv")
    df_combined.to_csv(csv_path, index=False)
    print(f"[+] Saved: {csv_path}")
    
    # Excel format
    excel_path = os.path.join(output_dir, "groundwater_predictions.xlsx")
    try:
        df_combined.to_excel(excel_path, index=False)
        print(f"[+] Saved: {excel_path}")
    except:
        print("[!] Excel save failed (openpyxl not installed)")
    
    return df_combined

def cnn_lstm_predict_with_rainfall(yearly_avg, rainfall_data, future_year, seq_len=3, output_dir="outputs"):
    """Train and predict with comprehensive visualizations"""
    
    X, y, n_features = prepare_lstm_data_with_rainfall(yearly_avg, rainfall_data, seq_len)
    
    if len(X) == 0:
        print("Warning: Not enough data for training")
        return None, None, None, []
    
    scaler_water = MinMaxScaler()
    scaler_rainfall = MinMaxScaler() if n_features > 1 else None
    
    X_scaled = X.copy().astype(float)
    X_scaled[:, :, 0] = scaler_water.fit_transform(X[:, :, 0].reshape(-1, 1)).reshape(X[:, :, 0].shape)
    
    if n_features > 1:
        X_scaled[:, :, 1] = scaler_rainfall.fit_transform(X[:, :, 1].reshape(-1, 1)).reshape(X[:, :, 1].shape)

    model = build_cnn_lstm_multifeature(seq_len, n_features)
    history = model.fit(X_scaled, y, epochs=100, batch_size=4, verbose=0)

    # Training predictions
    y_pred_train = model.predict(X_scaled, verbose=0)
    mse = mean_squared_error(y, y_pred_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)

    print("\n[*] TRAINING ACCURACY (With Rainfall Features)")
    print(f"    MSE  : {mse:.4f}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R² : {r2:.4f}")
    print(f"    Features: {n_features}")

    # ----- VISUALIZATIONS -----
    os.makedirs(output_dir, exist_ok=True)

    # 1. Training loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], linewidth=2)
    plt.title("CNN + LSTM Training Loss (With Rainfall)", fontsize=13, fontweight='bold')
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel("Loss (MSE)", fontsize=11)
    plt.grid(True, alpha=0.3)
    loss_plot_path = os.path.join(output_dir, "00_training_loss.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {loss_plot_path}")

    # 2. Actual vs Predicted
    plot_actual_vs_predicted(yearly_avg, y_pred_train, output_dir)

    # 3. Scatter plot
    plot_scatter_actual_vs_predicted(yearly_avg, y_pred_train, output_dir)

    # 4. Residuals
    plot_residuals_boxplot(yearly_avg, y_pred_train, output_dir)

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
        
        if n_features > 1:
            estimated_rainfall = np.mean(last_seq[1:, 1])
            last_seq = np.vstack([last_seq[1:], [next_pred, estimated_rainfall]])
        else:
            last_seq = np.vstack([last_seq[1:], [[next_pred]]])

    # 5. Future trend plot
    plot_future_trend(yearly_avg, future_predictions, output_dir)

    # 6. Save prediction data
    prediction_data = save_predictions_data(future_predictions, yearly_avg, output_dir)

    return future_predictions, loss_plot_path, (mse, rmse, mae, r2), prediction_data

def predict_until_year_with_rainfall(latitude, longitude, district, future_year, model, df_long, rainfall_df, output_dir="outputs"):
    """Main prediction function with enhanced outputs"""
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

    print(f"\n[*] Historical data for well {nearest_well}:")
    print(f"    Time period: {yearly_avg['Year'].min()} - {yearly_avg['Year'].max()}")
    print(f"    Records: {len(yearly_avg)}")

    rainfall_yearly = None
    if district and rainfall_df is not None:
        rainfall_data = get_rainfall_for_district(district, rainfall_df)
        if rainfall_data is not None:
            rainfall_yearly = pd.Series([rainfall_data['ANNUAL']] * len(yearly_avg), index=yearly_avg.index)
            print(f"[+] Using annual rainfall data for district: {district}")
            print(f"    Annual rainfall: {rainfall_data['ANNUAL']:.2f} mm")
        else:
            print(f"[-] District '{district}' not found in rainfall database")

    predictions, loss_img, scores, pred_data = cnn_lstm_predict_with_rainfall(yearly_avg, rainfall_yearly, future_year)

    if predictions is None:
        return nearest_well, {}, "", "", scores, None

    return nearest_well, predictions, loss_img, "", scores, pred_data

# Main execution
if __name__ == "__main__":
    print("="*70)
    print("GROUNDWATER LEVEL PREDICTION WITH ENHANCED VISUALIZATIONS")
    print("="*70)
    
    spatial_model, df_long = create_spatial_model('CGWB_data_main_cleaned.csv')
    rainfall_df = load_rainfall_data('district wise rainfall normal.csv')

    if spatial_model:
        print("\n[?] Enter location data:")
        lat = float(input("    Latitude: "))
        lon = float(input("    Longitude: "))
        district = input("    District: ").strip()
        year = int(input("    Year: "))

        print("\n[*] Processing... generating visualizations and predictions")
        well, preds, loss_file, _, acc, pred_data = predict_until_year_with_rainfall(
            lat, lon, district, year, spatial_model, df_long, rainfall_df
        )

        print("\n" + "="*70)
        print("PREDICTIONS & RESULTS")
        print("="*70)
        print(f"\n[*] Well: {well}")
        print(f"[*] District: {district}\n")

        print("[*] Year-wise Predictions:")
        for yr, lvl in preds.items():
            print(f"    {yr}: {lvl:.2f} m")

        print(f"\n[*] Model Performance:")
        print(f"    MSE  : {acc[0]:.4f}")
        print(f"    RMSE : {acc[1]:.4f}")
        print(f"    MAE  : {acc[2]:.4f}")
        print(f"    R²   : {acc[3]:.4f}")

        print(f"\n[*] Generated Files (in outputs/):")
        print(f"    - 00_training_loss.png")
        print(f"    - 01_actual_vs_predicted.png")
        print(f"    - 02_scatter_actual_vs_predicted.png")
        print(f"    - 03_residuals_boxplot.png")
        print(f"    - 04_future_trend.png")
        print(f"    - groundwater_predictions.csv")
        
        print("\n" + "="*70)
        print("[+] All visualizations and data exported successfully!")
        print("="*70)
