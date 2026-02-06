# 📦 Demand Forecaster

A machine learning project for retail inventory demand forecasting using Python and advanced regression models.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 🎯 Overview

This project predicts sales demand for products across multiple stores using various ML regression models. It helps retailers optimize inventory levels, reduce stockouts, and minimize overstock situations.

## 📊 Dataset

The dataset (`StoreDemand.csv`) contains:

| Feature | Description |
|---------|-------------|
| `date` | Transaction date (2021-2026) |
| `store` | Store ID (1-15) |
| `store_city` | City where store is located |
| `store_region` | Geographic region (North, South, East, West, Central) |
| `item` | Product ID (1-75) |
| `category` | Product category |
| `unit_price` | Price per unit |
| `sales` | Number of units sold |
| `stock_level` | Current inventory level |
| `promotion` | Whether item was on promotion (0/1) |
| `weather` | Weather condition |
| `revenue` | Total revenue |

**Dataset Statistics:**
- 📍 **15 unique stores** across India
- 🛍️ **75 different products** in 5 categories
- 📅 **5+ years** of sales data (2021-2026)
- 📈 **2+ million records**

### 🔧 Generate Your Own Dataset

Since the dataset is too large for GitHub, you can generate it yourself using the provided script:

```bash
python generate_data.py
```

This will create a `StoreDemand.csv` file with:
- Realistic sales patterns with seasonal variations
- Festive season spikes (October-December)
- Weekend vs weekday differences
- Weather impact on sales
- Promotional effects

> **Note**: Generation takes a few minutes due to the large dataset size (~2M records).

## ✨ Features

### 1. Feature Engineering
- 📆 Date extraction (year, month, day)
- 🗓️ Weekend indicator (weekday vs weekend)
- 🎉 Holiday indicator (Indian public holidays)
- 🔄 Cyclical month encoding (sin/cos transformation)
- 📊 Categorical encoding (store city, region, category, weather)

### 2. Exploratory Data Analysis
- 📊 Sales analysis by store, year, month, weekday
- 📈 30-day Simple Moving Average visualization
- 📉 Distribution and outlier detection
- 🔗 Correlation heatmap analysis

### 3. Models Implemented
| Model | Description |
|-------|-------------|
| **Linear Regression** | Baseline linear model |
| **XGBoost Regressor** | Gradient boosting (Best performer ⭐) |
| **Lasso Regression** | L1 regularized linear model |
| **Ridge Regression** | L2 regularized linear model |

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hardik0385/DemandForcaster.git
   cd DemandForcaster
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   .\env\Scripts\activate  # Windows
   source env/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate the dataset**
   ```bash
   python generate_data.py
   ```

## 💻 Usage

```bash
python demand_forecasting.py
```

The script will:
1. Load and analyze the dataset
2. Perform feature engineering
3. Generate visualizations (saved as PNG files)
4. Train multiple ML models
5. Display model comparison results

## 📈 Results

| Model | Training MAE | Validation MAE |
|-------|--------------|----------------|
| Linear Regression | 7.46 | 7.45 |
| **XGBoost Regressor** | **1.45** | **1.47** ⭐ |
| Lasso Regression | 9.56 | 9.63 |
| Ridge Regression | 7.46 | 7.45 |

**🏆 Winner: XGBoost Regressor** with the lowest validation MAE of 1.47

## 📁 Generated Visualizations

| File | Description |
|------|-------------|
| `feature_analysis.png` | Sales breakdown by store, year, month, weekday |
| `sales_by_day.png` | Daily sales trend analysis |
| `sma_analysis.png` | 30-day Simple Moving Average |
| `sales_distribution.png` | Sales distribution & outlier detection |
| `correlation_heatmap.png` | Feature correlation matrix |

## 📁 Project Structure

```
DemandForecaster/
├── demand_forecasting.py   # Main ML training script
├── generate_data.py        # Dataset generator script
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── LICENSE                # MIT License
└── .gitignore             # Git ignore rules
```

## 📋 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
holidays
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on concepts from [GeeksforGeeks - Inventory Demand Forecasting](https://www.geeksforgeeks.org/machine-learning/inventory-demand-forecasting-using-machine-learning-python/)

---

<p align="center">Made with ❤️ by Hardik Agrawal</p>
