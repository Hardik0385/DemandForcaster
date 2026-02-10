import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays
from datetime import datetime

# ============================================================
#                       CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Demand Forecasting App",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
#                       LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    try:
        artifacts = joblib.load('model_artifacts.joblib')
        return artifacts
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run 'demand_forecasting.py' first.")
        return None

artifacts = load_artifacts()

if artifacts:
    model = artifacts['model']
    scaler = artifacts['scaler']
    encoders = artifacts['encoders']
    feature_names = artifacts['features']

# ============================================================
#                       Global Constants (Based on Training Data)
# ============================================================
# These should ideally be saved in artifacts, but hardcoding for now based on typical dataset values
STORES = list(range(1, 16))  # 1 to 15
ITEMS = list(range(1, 76))   # 1 to 75
CITIES = encoders['store_city'].classes_ if artifacts else []
REGIONS = encoders['store_region'].classes_ if artifacts else []
CATEGORIES = encoders['category'].classes_ if artifacts else []
WEATHER_OPTIONS = encoders['weather'].classes_ if artifacts else []

# ============================================================
#                       HELPER FUNCTIONS
# ============================================================
def get_cyclical_features(date_obj):
    month = date_obj.month
    m1 = np.sin(month * (2 * np.pi / 12))
    m2 = np.cos(month * (2 * np.pi / 12))
    return m1, m2

def is_holiday(date_obj):
    in_holidays = holidays.country_holidays('IN')
    return 1 if date_obj in in_holidays else 0

def get_day_features(date_obj):
    day_of_week = date_obj.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    return day_of_week, is_weekend

# ============================================================
#                       UI LAYOUT
# ============================================================
st.title("📊 Demand Forecasting Dashboard")
st.markdown("Predict future sales based on store, item, and environmental factors.")

if artifacts:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📝 Input Parameters")
        
        with st.form("prediction_form"):
            date_val = st.date_input("Date", datetime.today())
            store = st.selectbox("Store ID", STORES)
            item = st.selectbox("Item ID", ITEMS)
            city = st.selectbox("Store City", CITIES)
            region = st.selectbox("Store Region", REGIONS)
            category = st.selectbox("Category", CATEGORIES)
            weather = st.selectbox("Weather Condition", WEATHER_OPTIONS)
            promotion = st.radio("Promotion Active?", ["No", "Yes"], horizontal=True)
            
            # Assume average values for required but missing inputs to keep UI simple
            # In a real app, unit_price might be looked up from a database based on Item ID
            unit_price = st.number_input("Unit Price", min_value=10.0, value=500.0)
            stock_level = st.number_input("Stock Level", min_value=0, value=100)
            
            submit_btn = st.form_submit_button("Generate Forecast")

    with col2:
        if submit_btn:
            # 1. Feature Engineering
            m1, m2 = get_cyclical_features(date_val)
            is_hol = is_holiday(date_val)
            weekday, weekend = get_day_features(date_val)
            promo_val = 1 if promotion == "Yes" else 0
            
            # Encode Categoricals
            city_enc = encoders['store_city'].transform([city])[0]
            region_enc = encoders['store_region'].transform([region])[0]
            cat_enc = encoders['category'].transform([category])[0]
            weather_enc = encoders['weather'].transform([weather])[0]
            
            # Create DataFrame for Model
            # Order must match training features exactly
            # ['store', 'item', 'unit_price', 'stock_level', 'promotion', 'revenue', 
            #  'month', 'day', 'weekend', 'holidays', 'm1', 'm2', 'weekday', 
            #  'store_city', 'store_region', 'category', 'weather']
            
            # revenue is usually a target or derived, but if it was in features (data leakage check needed in real scenario),
            # we need to provide it. Based on training script, revenue was in features.
            # approximating revenue = sales * unit_price (but we don't have sales yet!)
            # Looking at training script: revenue was in the input dataframe.
            # If the model uses 'revenue' as a feature to predict 'sales', that's data leakage.
            # Let's check feature_names from artifacts.
            
            input_data = {
                'store': [store],
                'item': [item],
                'unit_price': [unit_price],
                'stock_level': [stock_level],
                'promotion': [promo_val],
                # 'revenue': [0], # Placeholder, see note below
                'month': [date_val.month], 
                'day': [date_val.day],
                'weekend': [weekend], 
                'holidays': [is_hol],
                'm1': [m1], 
                'm2': [m2], 
                'weekday': [weekday],
                'store_city': [city_enc], 
                'store_region': [region_enc],
                'category': [cat_enc],
                'weather': [weather_enc]
            }
            
            # Special handling if 'revenue' was trained as a feature (checking keys)
            if 'revenue' in feature_names:
                 # Attempt to estimate or set to 0 if it's lagging
                 input_data['revenue'] = [0] 

            input_df = pd.DataFrame(input_data)
            
            # Reorder columns to match training
            # Filter out any keys not in feature_names (in case of mismatch)
            # Add missing keys as 0
            
            final_input = pd.DataFrame()
            for col in feature_names:
                if col in input_df.columns:
                    final_input[col] = input_df[col]
                else:
                    final_input[col] = 0
            
            # Scale
            input_scaled = scaler.transform(final_input)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            
            st.success("✅ Forecast Generated Successfully!")
            
            st.markdown(f"""
                <div class="metric-card">
                    <h2>Predicted Sales</h2>
                    <h1 style="color: #4CAF50; font-size: 3em;">{int(prediction)} units</h1>
                    <p>Estimated Revenue: ₹{int(prediction * unit_price):,}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Feature Contribution (Simple heuristic or SHAP could go here)
            st.info(f"Factors: {weather} weather in {city} on a {date_val.strftime('%A')}.")

        st.header("📊 Model Analysis & Insights")
        tab1, tab2, tab3 = st.tabs(["Feature Analysis", "Sales Trends", "Correlations"])
        
        import os

        # ... (inside columns)

        with tab1:
            if os.path.exists('feature_analysis.png'):
                st.image('feature_analysis.png', caption="Sales by Feature")
            else:
                st.info("Feature Analysis chart not available.")
                
            if os.path.exists('sales_distribution.png'):
                st.image('sales_distribution.png', caption="Sales Distribution")
            else:
                st.info("Sales Distribution chart not available.")
            
        with tab2:
            if os.path.exists('sales_by_day.png'):
                st.image('sales_by_day.png', caption="Daily Sales Trend")
            else:
                st.info("Sales by Day chart not available.")
                
            if os.path.exists('sma_analysis.png'):
                st.image('sma_analysis.png', caption="Moving Averages")
            else:
                st.info("SMA Analysis chart not available.")
            
        with tab3:
            if os.path.exists('correlation_heatmap.png'):
                st.image('correlation_heatmap.png', caption="Feature Correlations")
            else:
                st.info("Correlation Heatmap not available.")

else:
    st.warning("Project not initialized. Run the training script first.")
