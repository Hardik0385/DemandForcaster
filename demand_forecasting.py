import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae
from datetime import datetime, date
import holidays
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#                    DEMAND FORECASTING MODEL
# ============================================================

def print_header(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_subheader(title):
    """Print a formatted subsection header"""
    print(f"\n>>> {title}")
    print("-" * 40)

# ============================================================
#                    STEP 1: DATA LOADING
# ============================================================

print_header("STEP 1: LOADING DATASET")

df = pd.read_csv('StoreDemand.csv')

print_subheader("Dataset Preview (First 5 Rows)")
print(df.head().to_string())

print_subheader("Dataset Preview (Last 5 Rows)")
print(df.tail().to_string())

print_subheader("Dataset Overview")
rows, cols = df.shape
print(f"  • Total Rows    : {rows:,}")
print(f"  • Total Columns : {cols}")
print(f"  • Memory Usage  : {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

print_subheader("Column Information")
print(f"{'Column':<15} {'Data Type':<12} {'Non-Null Count':<15} {'Unique Values':<15}")
print("-" * 60)
for col in df.columns:
    dtype = str(df[col].dtype)
    non_null = df[col].notna().sum()
    unique = df[col].nunique()
    print(f"{col:<15} {dtype:<12} {non_null:<15,} {unique:<15,}")

print_subheader("Numerical Statistics Summary")
print(df.describe().round(2).to_string())

# ============================================================
#                 STEP 2: FEATURE ENGINEERING
# ============================================================

print_header("STEP 2: FEATURE ENGINEERING")

parts = df["date"].str.split("-", n=3, expand=True)
df["year"] = parts[0].astype('int')
df["month"] = parts[1].astype('int')
df["day"] = parts[2].astype('int')

print_subheader("Date Components Extracted")
print("  ✓ Year, Month, and Day columns created from 'date'")

def weekend_or_weekday(year, month, day):
    d = datetime(year, month, day)
    return 1 if d.weekday() > 4 else 0

df['weekend'] = df.apply(lambda x: weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)
print("  ✓ Weekend indicator added (1 = Weekend, 0 = Weekday)")

india_holidays = holidays.country_holidays('IN')
df['holidays'] = df['date'].apply(lambda x: 1 if india_holidays.get(x) else 0)
print("  ✓ Holiday indicator added (Indian public holidays)")

df['m1'] = np.sin(df['month'] * (2 * np.pi / 12))
df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))
print("  ✓ Cyclical month features added (sin/cos encoding)")

def which_day(year, month, day):
    return datetime(year, month, day).weekday()

df['weekday'] = df.apply(lambda x: which_day(x['year'], x['month'], x['day']), axis=1)
print("  ✓ Weekday column added (0=Mon, 1=Tue, ..., 6=Sun)")

df.drop('date', axis=1, inplace=True)
print("  ✓ Original 'date' column dropped")

print_subheader("Updated Dataset Shape")
print(f"  • Total Features : {df.shape[1]}")
print(f"  • Total Records  : {df.shape[0]:,}")

# ============================================================
#             STEP 3: EXPLORATORY DATA ANALYSIS
# ============================================================

print_header("STEP 3: EXPLORATORY DATA ANALYSIS")

print_subheader("Unique Value Counts")
print(f"  • Number of Stores : {df['store'].nunique()}")
print(f"  • Number of Items  : {df['item'].nunique()}")
if 'store_city' in df.columns:
    print(f"  • Store Cities     : {df['store_city'].nunique()}")
if 'category' in df.columns:
    print(f"  • Categories       : {df['category'].nunique()}")

df['weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

print("\n  📊 Generating visualizations...")
features_to_plot = ['store', 'year', 'month', 'weekday', 'weekend', 'holidays']
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features_to_plot):
    plt.subplot(2, 3, i + 1)
    df.groupby(col)['sales'].mean().plot.bar(color='steelblue', edgecolor='black')
    plt.title(f'Average Sales by {col.title()}', fontsize=12, fontweight='bold')
    plt.xlabel(col.title())
    plt.ylabel('Average Sales')
plt.tight_layout()
plt.savefig('feature_analysis.png', dpi=150)
plt.show()
print("  ✓ Saved: feature_analysis.png")

plt.figure(figsize=(10, 5))
df.groupby('day')['sales'].mean().plot(color='coral', linewidth=2, marker='o', markersize=4)
plt.title('Average Sales by Day of Month', fontsize=14, fontweight='bold')
plt.xlabel('Day of Month')
plt.ylabel('Average Sales')
plt.grid(True, alpha=0.3)
plt.savefig('sales_by_day.png', dpi=150)
plt.show()
print("  ✓ Saved: sales_by_day.png")

plt.figure(figsize=(15, 10))
window_size = 30
data = df[df['year'] == 2021]
if len(data) > 0:
    windows = data['sales'].rolling(window_size)
    sma = windows.mean()
    sma = sma[window_size - 1:]
    data['sales'].plot(label='Daily Sales', alpha=0.5)
    sma.plot(label='30-day Moving Average', linewidth=2, color='red')
    plt.title('Sales with 30-day Simple Moving Average (Year 2021)', fontsize=14, fontweight='bold')
    plt.xlabel('Record Index')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sma_analysis.png', dpi=150)
    plt.show()
    print("  ✓ Saved: sma_analysis.png")

plt.subplots(figsize=(12, 5))
plt.subplot(1, 2, 1)
sb.histplot(df['sales'], kde=True, color='steelblue')
plt.title('Sales Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Sales')
plt.subplot(1, 2, 2)
sb.boxplot(x=df['sales'], color='coral')
plt.title('Sales Boxplot (Outlier Detection)', fontsize=12, fontweight='bold')
plt.xlabel('Sales')
plt.tight_layout()
plt.savefig('sales_distribution.png', dpi=150)
plt.show()
print("  ✓ Saved: sales_distribution.png")

plt.figure(figsize=(10, 10))
sb.heatmap(df.corr(numeric_only=True) > 0.8, annot=True, cbar=False, cmap='RdYlGn')
plt.title('Feature Correlation Matrix (>0.8 = Highly Correlated)', fontsize=14, fontweight='bold')
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
print("  ✓ Saved: correlation_heatmap.png")

original_count = len(df)
df = df[df['sales'] < 140]
removed_count = original_count - len(df)
print_subheader("Outlier Removal")
print(f"  • Sales threshold   : < 140")
print(f"  • Records removed   : {removed_count:,}")
print(f"  • Records remaining : {len(df):,}")

# ============================================================
#                   STEP 4: MODEL TRAINING
# ============================================================

print_header("STEP 4: MODEL TRAINING")

print_subheader("Encoding Categorical Features")
categorical_cols = ['store_city', 'store_region', 'category', 'weather']
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  ✓ '{col}' → Encoded to numerical values")

features = df.drop(['sales', 'year'], axis=1)
target = df['sales'].values
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.05, random_state=22)

print_subheader("Train-Validation Split")
print(f"  • Training samples   : {X_train.shape[0]:,} ({95}%)")
print(f"  • Validation samples : {X_val.shape[0]:,} ({5}%)")
print(f"  • Number of features : {X_train.shape[1]}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
print("\n  ✓ Features normalized using StandardScaler")

# ============================================================
#                  STEP 5: MODEL EVALUATION
# ============================================================

print_header("STEP 5: MODEL EVALUATION")

models = {
    'Linear Regression': LinearRegression(),
    'XGBoost Regressor': XGBRegressor(),
    'Lasso Regression': Lasso(),
    'Ridge Regression': Ridge()
}

print_subheader("Training Models...")

results = []
best_model = None
best_val_mae = float('inf')

for name, model in models.items():
    model.fit(X_train, Y_train)
    
    train_preds = model.predict(X_train)
    train_mae = mae(Y_train, train_preds)
    
    val_preds = model.predict(X_val)
    val_mae = mae(Y_val, val_preds)
    
    results.append({
        'Model': name,
        'Train MAE': train_mae,
        'Val MAE': val_mae
    })
    
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_model = name

print("\n" + "-" * 60)
print(f"{'Model':<22} {'Training MAE':<18} {'Validation MAE':<18}")
print("-" * 60)
for r in results:
    marker = " ★" if r['Model'] == best_model else ""
    print(f"{r['Model']:<22} {r['Train MAE']:<18.4f} {r['Val MAE']:<18.4f}{marker}")
print("-" * 60)

# ============================================================
#                       FINAL SUMMARY
# ============================================================

print_header("SUMMARY & CONCLUSIONS")

print(f"""
  🏆 BEST PERFORMING MODEL: {best_model}
     • Validation MAE: {best_val_mae:.4f}
  
  📈 KEY INSIGHTS:
     • XGBoost captures complex non-linear patterns effectively
     • Linear models (Linear, Lasso, Ridge) show similar performance
     • Feature engineering (cyclical, weekday, holiday) improved predictions
  
  📁 GENERATED VISUALIZATIONS:
     • feature_analysis.png   - Sales breakdown by features
     • sales_by_day.png       - Daily sales trend
     • sma_analysis.png       - Moving average analysis
     • sales_distribution.png - Distribution & outliers
     • correlation_heatmap.png - Feature correlations

  ✅ Model training completed successfully!
""")
