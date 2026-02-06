"""
Data Generator for Demand Forecasting Project
==============================================
This script generates a synthetic StoreDemand.csv dataset for the demand forecasting model.
Run this script first if you don't have the dataset.

Usage:
    python generate_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_store_demand_data(
    start_date='2021-01-01',
    end_date='2026-02-06',
    num_stores=15,
    num_items=75
):
    """
    Generate synthetic store demand data.
    
    Parameters:
    -----------
    start_date : str
        Start date for the dataset
    end_date : str
        End date for the dataset
    num_stores : int
        Number of stores (default: 15)
    num_items : int
        Number of items/products (default: 75)
    
    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    print("=" * 60)
    print("  GENERATING STORE DEMAND DATASET")
    print("=" * 60)
    store_cities = [
        'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata',
        'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
        'Chandigarh', 'Bhopal', 'Indore', 'Nagpur', 'Patna'
    ]
    
    store_regions = {
        'Mumbai': 'West', 'Pune': 'West', 'Ahmedabad': 'West',
        'Delhi': 'North', 'Jaipur': 'North', 'Lucknow': 'North', 
        'Chandigarh': 'North', 'Patna': 'North',
        'Bangalore': 'South', 'Chennai': 'South', 'Hyderabad': 'South',
        'Kolkata': 'East',
        'Bhopal': 'Central', 'Indore': 'Central', 'Nagpur': 'Central'
    }
    categories = [
        'Electronics', 'Clothing', 'Food & Beverages', 
        'Home & Kitchen', 'Beauty & Personal Care'
    ]
    weather_conditions = ['Sunny', 'Rainy', 'Cloudy', 'Hot', 'Cold']
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = pd.date_range(start=start, end=end, freq='D')
    print(f"\n  📅 Date range: {start_date} to {end_date}")
    print(f"  🏪 Stores: {num_stores}")
    print(f"  📦 Products: {num_items}")
    print(f"  📊 Expected records: {len(dates) * num_stores * num_items:,}")
    np.random.seed(42)
    item_categories = {i: categories[(i-1) % len(categories)] for i in range(1, num_items + 1)}
    item_prices = {i: round(np.random.uniform(23, 4515), 2) for i in range(1, num_items + 1)}
    print("\n  ⏳ Generating data... (this may take a few minutes)")
    records = []
    total_combinations = len(dates) * num_stores * num_items
    progress_step = total_combinations // 10
    
    idx = 0
    for date in dates:
        month = date.month
        day_of_week = date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        seasonal_mult = 1.0
        if month in [10, 11, 12]: 
            seasonal_mult = 1.3
        elif month in [1, 2]:  
            seasonal_mult = 0.85
        elif month in [6, 7, 8]:  
            seasonal_mult = 0.9
        weekend_mult = 1.15 if is_weekend else 1.0
        for store in range(1, num_stores + 1):
            city = store_cities[store - 1]
            region = store_regions[city]
            store_factor = 0.8 + (store / num_stores) * 0.4
            weather = random.choice(weather_conditions)
            weather_mult = 1.0
            if weather == 'Rainy':
                weather_mult = 0.85
            elif weather == 'Hot':
                weather_mult = 0.9
            elif weather == 'Cold':
                weather_mult = 0.95
            
            for item in range(1, num_items + 1):
                # Base sales with randomness
                base_sales = np.random.poisson(250)
                
                # Apply all multipliers
                sales = int(base_sales * seasonal_mult * weekend_mult * store_factor * weather_mult)
                sales = max(31, min(sales, 2645))  # Clamp to realistic range
                
                # Promotion (10% of records have promotions)
                promotion = 1 if np.random.random() < 0.1 else 0
                if promotion:
                    sales = int(sales * 1.25)
                
                # Stock level
                stock_level = np.random.randint(36, 4193)
                
                # Calculate revenue
                unit_price = item_prices[item]
                revenue = round(sales * unit_price, 2)
                
                records.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'store': store,
                    'store_city': city,
                    'store_region': region,
                    'item': item,
                    'category': item_categories[item],
                    'unit_price': unit_price,
                    'sales': sales,
                    'stock_level': stock_level,
                    'promotion': promotion,
                    'weather': weather,
                    'revenue': revenue
                })
                
                idx += 1
                if idx % progress_step == 0:
                    pct = (idx / total_combinations) * 100
                    print(f"     Progress: {pct:.0f}%")
    
    df = pd.DataFrame(records)
    
    print("\n  ✅ Data generation complete!")
    print(f"\n  📊 Dataset Statistics:")
    print(f"     • Total records: {len(df):,}")
    print(f"     • Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"     • Stores: {df['store'].nunique()}")
    print(f"     • Items: {df['item'].nunique()}")
    print(f"     • Categories: {df['category'].nunique()}")
    
    return df


def main():

    df = generate_store_demand_data()
    
    output_file = 'StoreDemand.csv'
    print(f"\n  💾 Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    file_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"  ✅ Saved successfully!")
    print(f"     • File: {output_file}")
    print(f"     • Memory size: {file_size_mb:.2f} MB")
    
    print("\n" + "=" * 60)
    print("  Dataset generated! You can now run:")
    print("  python demand_forecasting.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
