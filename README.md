# Used Car Price Analysis

## Project Overview

This project analyzes a dataset of 426,880 used cars from Kaggle to understand what factors drive car prices. The analysis follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework and provides actionable recommendations for a used car dealership.

## Business Problem

A used car dealership needs to understand what factors influence used car prices to optimize their inventory and pricing strategy. The goal is to identify key price drivers and provide data-driven recommendations for business decisions.

## Dataset

- **Source**: Kaggle Used Cars Dataset
- **Size**: 426,880 records, 18 features
- **Features**: Price, year, manufacturer, model, condition, odometer, fuel type, transmission, etc.
- **Target Variable**: Price

## Key Findings

### Model Performance
- **Best Model**: Random Forest Regressor
- **R² Score**: 0.85 (explains 85% of price variation)
- **RMSE**: $3,200 (average prediction error)
- **Cross-validation**: Confirms model reliability

### Most Important Price Drivers
1. **Year/Model Year** - Newer cars command significantly higher prices
2. **Odometer Reading** - Lower mileage vehicles have premium pricing
3. **Manufacturer** - Brand reputation significantly impacts pricing
4. **Vehicle Condition** - Excellent condition cars command 40%+ premium
5. **Fuel Type** - Electric and hybrid vehicles have higher prices

### Price Patterns
- Car age has -0.65 correlation with price (older = cheaper)
- Odometer has -0.45 correlation with price (higher mileage = cheaper)
- Luxury brands command 60%+ price premium
- SUVs and trucks have 15% price premium over sedans

## Recommendations for Used Car Dealership

### Inventory Strategy
1. **Focus on Good Condition Vehicles** - Best balance of price and demand
2. **Prioritize Low-Mileage Cars** - Significant price premium for well-maintained vehicles
3. **Target Luxury Brands** - Higher profit margins despite higher acquisition costs
4. **Consider Electric/Hybrid** - Growing market with premium pricing

### Pricing Strategy
1. **Use Data-Driven Pricing** - Implement predictive model for accurate pricing
2. **Factor in Vehicle History** - Clean titles and maintenance records add significant value
3. **Consider Market Timing** - Prices vary by season and market conditions
4. **Premium for Condition** - Excellent condition cars can command 40%+ premium

### Marketing Strategy
1. **Highlight Condition** - Emphasize vehicle condition in marketing materials
2. **Feature Low Mileage** - Promote low-mileage vehicles as premium options
3. **Leverage Brand Value** - Use manufacturer reputation in marketing
4. **Target Demographics** - Focus on customers valuing reliability and condition

## Analysis Notebook

**[View the complete analysis here](prompt_II.ipynb)**

The Jupyter notebook contains:
- **CRISP-DM Framework Implementation**
- **Comprehensive Data Exploration**
- **Multiple Regression Models** (Linear, Ridge, Lasso, Random Forest)
- **Cross-validation and Hyperparameter Tuning**
- **Feature Importance Analysis**
- **Business Insights and Recommendations**

## Technical Details

### Models Used
- Linear Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Random Forest Regressor

### Evaluation Metrics
- **R² Score**: Measures explained variance
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Cross-validation**: Ensures model reliability

### Feature Engineering
- Created age feature (2024 - year)
- Created price per mile feature
- Created luxury brand indicator
- Created SUV indicator
- Encoded categorical variables

## Next Steps

1. **Implement Real-Time Pricing Model** - Deploy the predictive model for live pricing
2. **Monitor Market Trends** - Track price changes and adjust inventory accordingly
3. **Expand Regional Analysis** - Include geographic factors in pricing
4. **Customer Segmentation** - Develop targeted marketing based on preferences
5. **Competitive Analysis** - Monitor competitor pricing strategies

## Success Metrics

- **Pricing Accuracy**: Reduce pricing errors by 30%
- **Inventory Turnover**: Increase by 20% through better pricing
- **Profit Margins**: Improve by 15% through optimized pricing
- **Customer Satisfaction**: Increase through fair and transparent pricing

## Requirements

To run this analysis, install the required packages:

```bash
pip install -r requirements.txt
```


## License

This project is for educational purposes as part of the UC Berkeley AIML curriculum.

---

*Analysis completed using CRISP-DM methodology with comprehensive data exploration, multiple regression modeling, and actionable business recommendations.*