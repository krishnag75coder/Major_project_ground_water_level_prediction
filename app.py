import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Page configuration
st.set_page_config(
    page_title="Groundwater Level Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    /* Main styling */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }
    
    h2 {
        color: #2c3e50;
        font-weight: 700;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
        margin-top: 2rem;
    }
    
    h3 {
        color: #667eea;
        font-weight: 600;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(102, 126, 234, 0.5);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e0f2f7 0%, #e8f5ff 100%);
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar-content {
        color: white;
    }
    
    /* Input styling */
    input[type="number"], input[type="text"] {
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    input[type="number"]:focus, input[type="text"]:focus {
        border-color: #764ba2 !important;
        box-shadow: 0 0 10px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #e0f2f7 0%, #e8f5ff 100%);
        border-radius: 8px;
        color: #667eea;
        font-weight: 600;
        padding: 12px 20px;
    }
    
    .stTabs [aria-selected="true"] [data-baseweb="tab"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(132, 250, 176, 0.4);
    }
    
    /* Markdown text styling */
    p {
        font-size: 16px;
        color: #2c3e50;
        line-height: 1.6;
    }
    
    /* Code styling */
    code {
        background: #f5f5f5;
        padding: 2px 6px;
        border-radius: 4px;
        color: #c7254e;
        font-weight: 500;
    }
    
    /* Metric value styling */
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: white;
    }
    
    .metric-label {
        font-size: 12px;
        opacity: 0.9;
        margin-top: 5px;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 5px;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    /* Table styling */
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 15px 0;
    }
    
    table th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    
    table td {
        padding: 10px 12px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    table tr:hover {
        background: #f5f7fa;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Section divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        margin: 30px 0;
        border-radius: 2px;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #667eea;
        margin: 0 3px;
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    @keyframes bounce {
        0%, 80%, 100% {
            opacity: 0.3;
            transform: scale(0);
        }
        40% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Card style */
    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
        border-left: 4px solid #667eea;
    }
    
    .card:hover {
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.2);
    }
    
    /* Label styling */
    label {
        color: #667eea;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<h1>💧 Groundwater Level Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #667eea; font-weight: 600;'> Predictions Using LSTM+CNN with Rainfall Integration</p>", unsafe_allow_html=True)
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Load data (cached)
@st.cache_resource
def load_data():
    try:
        df_cgwb = pd.read_csv('CGWB_data_main_cleaned.csv')
        df_rainfall = pd.read_csv('district wise rainfall normal.csv')
        return df_cgwb, df_rainfall
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def create_knn_model(df_cgwb):
    id_vars = ['STATE', 'DISTRICT', 'LAT', 'LON', 'SITE_TYPE', 'WLCODE']
    df_long = pd.melt(df_cgwb, id_vars=id_vars, var_name='Date', value_name='Water_Level')
    df_long['Date'] = pd.to_datetime(df_long['Date'], errors='coerce')
    df_long.dropna(subset=['Water_Level', 'LAT', 'LON', 'Date'], inplace=True)
    
    df_recent = df_long.loc[df_long.groupby('WLCODE')['Date'].idxmax()]
    X = df_recent[['LAT', 'LON']]
    y = df_recent['Water_Level']
    
    knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn_model.fit(X, y)
    
    return knn_model, df_long

# Load data
df_cgwb, df_rainfall = load_data()

if df_cgwb is None or df_rainfall is None:
    st.error("Could not load data files. Please ensure CGWB_data_main_cleaned.csv and district wise rainfall normal.csv are in the same directory.")
    st.stop()

knn_model, df_long = create_knn_model(df_cgwb)

# Input Parameters
col1, col2 = st.sidebar.columns(2)

with col1:
    latitude = st.number_input(
        "Latitude",
        min_value=8.0,
        max_value=35.0,
        value=28.5355,
        step=0.01,
        help="Enter latitude (8-35°N)"
    )

with col2:
    longitude = st.number_input(
        "Longitude",
        min_value=68.0,
        max_value=97.0,
        value=77.3910,
        step=0.01,
        help="Enter longitude (68-97°E)"
    )

district = st.sidebar.text_input(
    "District Name (Optional)",
    value="",
    help="Leave blank for auto-detection"
)

target_year = st.sidebar.slider(
    "Target Prediction Year",
    min_value=2025,
    max_value=2050,
    value=2035,
    step=1
)

# Run Prediction Button
if st.sidebar.button("🚀 Run Prediction", key="predict_btn"):
    
    with st.spinner("🔄 Processing... This may take 1-2 minutes"):
        try:
            # Find nearest well
            closest_wells = df_long.drop_duplicates(subset=['WLCODE'])
            distances = np.sqrt((closest_wells['LAT'] - latitude)**2 + (closest_wells['LON'] - longitude)**2)
            nearest_idx = distances.idxmin()
            well_id = closest_wells.loc[nearest_idx, 'WLCODE']
            pred_district = closest_wells.loc[nearest_idx, 'DISTRICT']
            state = closest_wells.loc[nearest_idx, 'STATE']
            
            # Get yearly data
            well_data = df_long[df_long['WLCODE'] == well_id].copy()
            well_data['Year'] = well_data['Date'].dt.year
            yearly_avg = well_data.groupby('Year')['Water_Level'].mean().sort_index()
            
            # Get rainfall
            district_str = str(pred_district).lower().strip()
            match = df_rainfall[df_rainfall['DISTRICT'].astype(str).str.lower().str.strip() == district_str]
            rainfall = float(match.iloc[0]['ANNUAL']) if not match.empty else 700
            
            # Prepare data
            X_data = yearly_avg.values.reshape(-1, 1)
            y_data = X_data[1:]
            X_data = X_data[:-1]
            
            scaler_water = MinMaxScaler()
            X_scaled = scaler_water.fit_transform(X_data)
            rainfall_val = rainfall / 1000.0
            X_dual = np.column_stack([X_scaled, np.ones(len(X_scaled)) * rainfall_val])
            
            seq_length = 3
            X_seq, y_seq, seq_years = [], [], []
            for i in range(len(X_dual) - seq_length):
                X_seq.append(X_dual[i:i+seq_length])
                y_seq.append(y_data[i+seq_length])
                seq_years.append(yearly_avg.index[i+seq_length])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq).reshape(-1, 1)
            y_seq_scaled = scaler_water.fit_transform(y_seq)
            
            # Build model
            model = Sequential([
                Conv1D(32, kernel_size=2, activation='relu', input_shape=(seq_length, 2)),
                MaxPooling1D(pool_size=1),
                LSTM(64, return_sequences=False),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            history = model.fit(X_seq, y_seq_scaled, epochs=100, batch_size=4, verbose=0)
            
            # Predictions
            y_pred = model.predict(X_seq, verbose=0)
            y_pred_actual = scaler_water.inverse_transform(y_pred).flatten()
            y_seq_actual = scaler_water.inverse_transform(y_seq_scaled).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_seq_actual, y_pred_actual)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_seq_actual, y_pred_actual)
            r2 = r2_score(y_seq_actual, y_pred_actual)
            
            # Get future predictions
            predictions = {}
            last_seq = X_dual[-seq_length:].copy()
            
            for year in range(yearly_avg.index.max() + 1, target_year + 1):
                pred_scaled = model.predict(last_seq.reshape(1, seq_length, 2), verbose=0)[0][0]
                pred_actual = scaler_water.inverse_transform([[pred_scaled]])[0][0]
                predictions[year] = pred_actual
                new_point = np.array([[pred_scaled, rainfall_val]])
                last_seq = np.vstack([last_seq[1:], new_point])
            
            # Store in session state
            st.session_state.prediction_complete = True
            st.session_state.well_id = well_id
            st.session_state.pred_district = pred_district
            st.session_state.state = state
            st.session_state.yearly_avg = yearly_avg
            st.session_state.rainfall = rainfall
            st.session_state.seq_years = seq_years
            st.session_state.y_seq_actual = y_seq_actual
            st.session_state.y_pred_actual = y_pred_actual
            st.session_state.history = history
            st.session_state.predictions = predictions
            st.session_state.target_year = target_year
            st.session_state.r2 = r2
            st.session_state.rmse = rmse
            st.session_state.mae = mae
            st.session_state.latitude = latitude
            st.session_state.longitude = longitude
            
            st.success("✅ Prediction completed successfully!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.stop()

# Display Results
if 'prediction_complete' in st.session_state and st.session_state.prediction_complete:
    
    # Location Information
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2>📍 Location Information</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏷️ Well ID", st.session_state.well_id)
    with col2:
        st.metric("🗺️ District", st.session_state.pred_district)
    with col3:
        st.metric("📍 State", st.session_state.state)
    with col4:
        st.metric("🌧️ Rainfall (mm)", f"{st.session_state.rainfall:.0f}")
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2>📊 Performance Metrics</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("✅ R² Score", f"{st.session_state.r2:.4f}", delta="Higher Better", delta_color="inverse")
    with col2:
        st.metric("📉 RMSE (m)", f"{st.session_state.rmse:.4f}", delta="Lower Better", delta_color="inverse")
    with col3:
        st.metric("📊 MAE (m)", f"{st.session_state.mae:.4f}", delta="Lower Better", delta_color="inverse")
    with col4:
        st.metric("📅 Years Predicted", f"{st.session_state.target_year - st.session_state.yearly_avg.index.max()}")
    
    # Visualizations
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2>📈 Visualizations</h2>", unsafe_allow_html=True)
    
    # Tab 1: Training Loss
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Training Loss", 
        "Actual vs Predicted", 
        "Scatter Plot", 
        "Residuals", 
        "Future Trend"
    ])
    
    with tab1:
        st.subheader("Training Loss Over Epochs")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(st.session_state.history.history['loss'], linewidth=2, color='#2E86AB')
        ax.set_title('Model Convergence', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Actual vs Predicted Groundwater Levels")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(st.session_state.seq_years, st.session_state.y_seq_actual, 'o-', label='Actual', linewidth=2, markersize=6, color='#A23B72')
        ax.plot(st.session_state.seq_years, st.session_state.y_pred_actual, 's--', label='Predicted', linewidth=2, markersize=5, color='#F18F01')
        ax.set_title('Training Period: Actual vs Predicted', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Water Level (m)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Accuracy Scatter Plot")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(st.session_state.y_seq_actual, st.session_state.y_pred_actual, s=100, alpha=0.6, color='#2E86AB', edgecolors='black', linewidth=1)
        min_val = min(st.session_state.y_seq_actual.min(), st.session_state.y_pred_actual.min())
        max_val = max(st.session_state.y_seq_actual.max(), st.session_state.y_pred_actual.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax.set_title('Actual vs Predicted Scatter', fontsize=14, fontweight='bold')
        ax.set_xlabel('Actual Water Level (m)', fontsize=12)
        ax.set_ylabel('Predicted Water Level (m)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with tab4:
        st.subheader("Residuals Analysis")
        residuals = st.session_state.y_seq_actual - st.session_state.y_pred_actual
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.boxplot(residuals, vert=True)
        ax1.set_ylabel('Residuals (m)', fontsize=12)
        ax1.set_title('Residuals Box Plot', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2.hist(residuals, bins=10, color='#A23B72', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residuals (m)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residuals Distribution', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
    
    with tab5:
        st.subheader("Future Groundwater Level Predictions")
        future_years = sorted(st.session_state.predictions.keys())
        future_values = [st.session_state.predictions[y] for y in future_years]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(st.session_state.yearly_avg.index, st.session_state.yearly_avg.values, 'o-', 
                label='Historical (Actual)', linewidth=2.5, markersize=7, color='#2E86AB')
        ax.plot(future_years, future_values, 's--', label='Future Predictions', 
                linewidth=2.5, markersize=6, color='#F18F01')
        ax.axvline(x=st.session_state.yearly_avg.index.max(), color='red', linestyle=':', 
                   linewidth=2, alpha=0.7, label='Prediction Start')
        ax.set_title(f'Groundwater Level Trend (Historical + Future to {st.session_state.target_year})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Water Level (m)', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Predictions Table
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2>📋 Year-wise Predictions</h2>", unsafe_allow_html=True)
    
    future_years = sorted(st.session_state.predictions.keys())
    future_values = [st.session_state.predictions[y] for y in future_years]
    
    df_results = pd.DataFrame({
        'Year': future_years,
        'Predicted_Water_Level_m': future_values,
        'Type': ['Future'] * len(future_years),
        'Well_ID': [st.session_state.well_id] * len(future_years),
        'District': [st.session_state.pred_district] * len(future_years),
        'Rainfall_mm': [st.session_state.rainfall] * len(future_years)
    })
    
    st.dataframe(df_results, use_container_width=True)
    
    # Download CSV
    csv = df_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv,
        file_name=f"predictions_{st.session_state.well_id}_{st.session_state.target_year}.csv",
        mime="text/csv"
    )
    
    # Summary Statistics
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2>📊 Prediction Summary</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 Historical Period", f"{st.session_state.yearly_avg.index.min()}-{st.session_state.yearly_avg.index.max()}")
    with col2:
        st.metric("💧 Last Recorded Level", f"{st.session_state.yearly_avg.iloc[-1]:.2f}m")
    with col3:
        st.metric("🎯 Predicted Target Level", f"{st.session_state.predictions[st.session_state.target_year]:.2f}m")
    
    # Information Box
    st.markdown("""
    <div class='info-box'>
        <h3 style='margin-top: 0; color: #667eea;'>✨ Model Information</h3>
        <ul style='color: #2c3e50;'>
            <li><strong>Architecture:</strong> CNN+LSTM with Rainfall Integration</li>
            <li><strong>Features:</strong> Water Level (time series) + Annual Rainfall</li>
            <li><strong>Training Window:</strong> 3-year sequences with independent feature scaling</li>
            <li><strong>Optimization:</strong> Adam optimizer with MSE loss function</li>
            <li><strong>Interpretation:</strong> R² > 0.55 indicates high confidence predictions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class='card'>
        <h2 style='color: #667eea; margin-top: 0;'>🎯 How to Use</h2>
        <div style='background: linear-gradient(135deg, #e0f2f7 0%, #e8f5ff 100%); padding: 15px; border-radius: 8px; margin: 15px 0;'>
            <h3 style='color: #667eea; margin-top: 0;'>Step 1: Configure Parameters</h3>
            <p style='color: #2c3e50; margin: 10px 0;'>
                <strong>In the sidebar, enter:</strong>
                <br>✓ Your latitude (8-35°N)
                <br>✓ Your longitude (68-97°E)
                <br>✓ District name (optional - auto-detected)
                <br>✓ Target prediction year (2025-2050)
            </p>
        </div>
        <div style='background: linear-gradient(135deg, #f5e0ff 0%, #e8e5ff 100%); padding: 15px; border-radius: 8px; margin: 15px 0;'>
            <h3 style='color: #764ba2; margin-top: 0;'>Step 2: Click "Run Prediction"</h3>
            <p style='color: #2c3e50; margin: 10px 0;'>
                The AI model will train and generate predictions
                <br>(Typical time: 1-2 minutes on first run)
            </p>
        </div>
        <div style='background: linear-gradient(135deg, #fff4e0 0%, #ffe8d0 100%); padding: 15px; border-radius: 8px; margin: 15px 0;'>
            <h3 style='color: #f59e0b; margin-top: 0;'>Step 3: View Results</h3>
            <p style='color: #2c3e50; margin: 10px 0;'>
                ✓ Performance metrics (R², RMSE, MAE)
                <br>✓ 5 detailed visualizations
                <br>✓ Year-wise prediction data
                <br>✓ Download predictions as CSV
            </p>
        </div>
    </div>
    
    <div class='section-divider'></div>
    
    <h2 style='color: #667eea;'>📍 Example Coordinates</h2>
    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
        <div class='card'>
            <strong style='color: #667eea;'>Delhi</strong><br>
            28.5355°N, 77.3910°E
        </div>
        <div class='card'>
            <strong style='color: #667eea;'>Mumbai</strong><br>
            19.0760°N, 72.8777°E
        </div>
        <div class='card'>
            <strong style='color: #667eea;'>Bangalore</strong><br>
            12.9716°N, 77.5946°E
        </div>
        <div class='card'>
            <strong style='color: #667eea;'>Kolkata</strong><br>
            22.5726°N, 88.3639°E
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; margin-top: 30px;'>
    <p style='color: #2c3e50; font-size: 14px; margin: 5px 0;'>
        <strong>💧 Groundwater Level Prediction System</strong>
    </p>
    <p style='color: #667eea; font-size: 12px; margin: 5px 0;'>
        AI-Powered | LSTM+CNN Architecture
    </p>
    <p style='color: #666; font-size: 11px; margin: 5px 0;'>
        Built with <span style='color: #E34C26;'>Streamlit</span> | <span style='color: #FF6F00;'>TensorFlow</span> | <span style='color: #F7931E;'>Scikit-learn</span>
    </p>
    <p style='color: #999; font-size: 10px; margin-top: 10px;'>
        © 2026 | For Water Resource Management | Real-time Predictions
    </p>
</div>
""", unsafe_allow_html=True)
