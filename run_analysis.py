#!/usr/bin/env python3
"""
Used Car Price Analysis Runner
==============================

This script runs the complete analysis and generates the Jupyter notebook
with all outputs, visualizations, and findings.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def main():
    """Run the complete analysis."""
    
    print("=== USED CAR PRICE ANALYSIS ===")
    print("Loading and analyzing dataset...")
    
    # Load the dataset
    df = pd.read_csv('data/vehicles.csv')
    
    print(f"Dataset loaded: {df.shape[0]:,} records, {df.shape[1]} features")
    
    # Data cleaning
    print("\n=== DATA CLEANING ===")
    df_clean = df.copy()
    
    # Remove unrealistic years
    df_clean = df_clean[(df_clean['year'] >= 1950) & (df_clean['year'] <= 2024)]
    print(f"After year filtering: {df_clean.shape[0]:,} records")
    
    # Remove extreme price outliers
    Q1_price = df_clean['price'].quantile(0.01)
    Q99_price = df_clean['price'].quantile(0.99)
    df_clean = df_clean[(df_clean['price'] >= Q1_price) & (df_clean['price'] <= Q99_price)]
    print(f"After price outlier removal: {df_clean.shape[0]:,} records")
    
    # Remove extreme odometer values
    df_clean = df_clean[df_clean['odometer'] <= 500000]
    print(f"After odometer filtering: {df_clean.shape[0]:,} records")
    
    # Handle missing values
    numerical_cols = ['price', 'year', 'odometer']
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    categorical_cols = ['manufacturer', 'condition', 'fuel', 'transmission', 'drive', 'type']
    for col in categorical_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
    
    # Feature engineering
    print("\n=== FEATURE ENGINEERING ===")
    df_clean['age'] = 2024 - df_clean['year']
    df_clean['price_per_mile'] = df_clean['price'] / (df_clean['odometer'] + 1)
    
    luxury_brands = ['bmw', 'mercedes-benz', 'audi', 'lexus', 'infiniti', 'acura', 'cadillac', 'lincoln']
    df_clean['is_luxury'] = df_clean['manufacturer'].str.lower().isin(luxury_brands).astype(int)
    
    suv_types = ['suv', 'truck', 'pickup']
    df_clean['is_suv'] = df_clean['type'].str.lower().isin(suv_types).astype(int)
    
    print(f"Final dataset shape: {df_clean.shape}")
    
    # Prepare features for modeling
    print("\n=== MODELING ===")
    feature_cols = ['year', 'odometer', 'age', 'price_per_mile', 'is_luxury', 'is_suv']
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[f'{col}_encoded'] = le.fit_transform(df_clean[col].astype(str))
            feature_cols.append(f'{col}_encoded')
    
    X = df_clean[feature_cols]
    y = df_clean['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'CV_R2_mean': cv_scores.mean(),
            'CV_R2_std': cv_scores.std(),
            'model': model
        }
        
        print(f"  R²: {r2:.4f}, RMSE: ${rmse:,.0f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['R2'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")
    
    # Generate visualizations
    print("\n=== GENERATING VISUALIZATIONS ===")
    
    # 1. Data exploration plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Used Car Price Analysis - Key Variables', fontsize=16, fontweight='bold')
    
    # Price distribution
    axes[0, 0].hist(df_clean['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].set_xlabel('Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_yscale('log')
    
    # Year distribution
    year_counts = df_clean['year'].value_counts().sort_index()
    axes[0, 1].plot(year_counts.index, year_counts.values, marker='o', linewidth=2)
    axes[0, 1].set_title('Cars by Year')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Odometer distribution
    axes[0, 2].hist(df_clean['odometer'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 2].set_title('Odometer Distribution')
    axes[0, 2].set_xlabel('Odometer (miles)')
    axes[0, 2].set_ylabel('Frequency')
    
    # Price vs Year
    axes[1, 0].scatter(df_clean['year'], df_clean['price'], alpha=0.3, s=1)
    axes[1, 0].set_title('Price vs Year')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Price ($)')
    
    # Price vs Odometer
    axes[1, 1].scatter(df_clean['odometer'], df_clean['price'], alpha=0.3, s=1)
    axes[1, 1].set_title('Price vs Odometer')
    axes[1, 1].set_xlabel('Odometer (miles)')
    axes[1, 1].set_ylabel('Price ($)')
    
    # Age distribution
    axes[1, 2].hist(df_clean['age'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 2].set_title('Car Age Distribution')
    axes[1, 2].set_xlabel('Age (years)')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Categorical analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Categorical Variables Analysis', fontsize=16, fontweight='bold')
    
    # Price by condition
    condition_price = df_clean.groupby('condition')['price'].mean().sort_values(ascending=False)
    axes[0, 0].bar(range(len(condition_price)), condition_price.values, color='lightcoral')
    axes[0, 0].set_title('Average Price by Condition')
    axes[0, 0].set_xlabel('Condition')
    axes[0, 0].set_ylabel('Average Price ($)')
    axes[0, 0].set_xticks(range(len(condition_price)))
    axes[0, 0].set_xticklabels(condition_price.index, rotation=45)
    
    # Price by fuel type
    fuel_price = df_clean.groupby('fuel')['price'].mean().sort_values(ascending=False)
    axes[0, 1].bar(range(len(fuel_price)), fuel_price.values, color='lightblue')
    axes[0, 1].set_title('Average Price by Fuel Type')
    axes[0, 1].set_xlabel('Fuel Type')
    axes[0, 1].set_ylabel('Average Price ($)')
    axes[0, 1].set_xticks(range(len(fuel_price)))
    axes[0, 1].set_xticklabels(fuel_price.index, rotation=45)
    
    # Top manufacturers
    manufacturer_price = df_clean.groupby('manufacturer')['price'].agg(['mean', 'count'])
    manufacturer_price = manufacturer_price[manufacturer_price['count'] >= 100].sort_values('mean', ascending=False).head(10)
    axes[1, 0].bar(range(len(manufacturer_price)), manufacturer_price['mean'].values, color='lightgreen')
    axes[1, 0].set_title('Top 10 Manufacturers by Average Price')
    axes[1, 0].set_xlabel('Manufacturer')
    axes[1, 0].set_ylabel('Average Price ($)')
    axes[1, 0].set_xticks(range(len(manufacturer_price)))
    axes[1, 0].set_xticklabels(manufacturer_price.index, rotation=45)
    
    # Price by transmission
    transmission_price = df_clean.groupby('transmission')['price'].mean().sort_values(ascending=False)
    axes[1, 1].bar(range(len(transmission_price)), transmission_price.values, color='gold')
    axes[1, 1].set_title('Average Price by Transmission')
    axes[1, 1].set_xlabel('Transmission')
    axes[1, 1].set_ylabel('Average Price ($)')
    axes[1, 1].set_xticks(range(len(transmission_price)))
    axes[1, 1].set_xticklabels(transmission_price.index, rotation=45)
    
    plt.tight_layout()
    plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Correlation matrix
    numerical_cols = ['price', 'year', 'odometer', 'age', 'price_per_mile', 'is_luxury', 'is_suv']
    correlation_matrix = df_clean[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Correlation Matrix - Numerical Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
        plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Generate insights
    print("\n=== KEY INSIGHTS ===")
    
    # Model performance
    print(f"Model Performance:")
    print(f"- Best Model: {best_model_name}")
    print(f"- R² Score: {results[best_model_name]['R2']:.4f}")
    print(f"- RMSE: ${results[best_model_name]['RMSE']:,.0f}")
    print(f"- MAE: ${results[best_model_name]['MAE']:,.0f}")
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nTop 5 Most Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"- {row['feature']}: {row['importance']:.4f}")
    
    # Price correlations
    age_price_corr = df_clean['age'].corr(df_clean['price'])
    odometer_price_corr = df_clean['odometer'].corr(df_clean['price'])
    
    print(f"\nKey Correlations:")
    print(f"- Age vs Price: {age_price_corr:.3f}")
    print(f"- Odometer vs Price: {odometer_price_corr:.3f}")
    
    # Price by condition
    condition_analysis = df_clean.groupby('condition')['price'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    print(f"\nAverage Price by Condition:")
    print(condition_analysis)
    
    # Luxury vs non-luxury
    luxury_analysis = df_clean.groupby('is_luxury')['price'].agg(['mean', 'count'])
    print(f"\nLuxury vs Non-Luxury Price Comparison:")
    print(luxury_analysis)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Generated files:")
    print("- data_exploration.png")
    print("- categorical_analysis.png") 
    print("- correlation_matrix.png")
    print("- feature_importance.png")
    
    return df_clean, results, best_model

if __name__ == "__main__":
    df_clean, results, best_model = main() 