# House Prices Prediction Using Regression Models
A machine learning project that predicts residential house prices in Ames, Iowa using advanced regression techniques and ensemble modeling.

## 📊 Project Overview
This project uses the Kaggle House Prices Dataset to build predictive models for estimating residential property sale prices. Through comprehensive exploratory data analysis, feature engineering, and model optimization, the project achieves robust prediction performance with an ensemble RMSE of 0.421 on test data.

## 🎯 Problem Statement
Predict the sale prices of residential homes in Ames, Iowa based on 79 explanatory variables describing various aspects of residential properties including:
- Physical attributes (size, quality, condition)
- Location characteristics (neighborhood, zoning)
- Amenities and features (garage, basement, pool)
- Sale conditions and timing
## 📁 Dataset
- Training Set: 1,460 samples with 81 features
- Test Set: 1,459 samples with 80 features
- Target Variable: SalePrice (residential property sale price in USD)
- Missing Values: 19 features contain missing data requiring imputation
## 🔍 Key Features
- Data Exploration & Analysis
- Comprehensive missing value analysis (19 features with nulls)
- Target variable transformation (log transformation reduced skewness from 1.88 to 0.12)
- Correlation analysis identifying key predictors (OverallQual: 70%+, GrLivArea: 70%+ correlation)
- ANOVA testing for categorical feature importance (Neighborhood, ExterQual, KitchenQual)
- Data Preprocessing
- Outlier removal (GrLivArea > 4000)
- Missing value imputation using mode/median strategies
- Label encoding for categorical variables
- Box-Cox transformation for skewness correction
- StandardScaler normalization for numeric features
- One-hot encoding for categorical features
## Feature Engineering
Created 25+ engineered features including:
- Binary presence indicators for property amenities
- Aggregated area calculations (total property area)
- Temporal features (years since remodeling)
- Neighborhood price bins based on median sale price
- Quality and condition composite scores
## Model Development
- XGBoost Regressor
- Lasso Regression
- Ensemble Model
- Hyperparameter tuning using GridSearchCV with cross-validation
## 🛠️ Technologies & Tools
**Programming Language:** Python\
**Machine Learning:** XGBoost, scikit-learn (Lasso Regression)\
**Data Processing:** pandas, NumPy\
**Feature Engineering:** Box-Cox transformation, StandardScaler, Label Encoder\
**Model Optimization:** GridSearchCV, cross-validation\
**Statistical Analysis:** Correlation analysis, ANOVA testing\
**Data Encoding:** One-hot encoding, label encoding
