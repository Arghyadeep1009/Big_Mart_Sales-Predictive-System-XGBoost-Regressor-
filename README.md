# Big Mart Sales Prediction

This project aims to predict the sales of items in different outlets of a big mart chain using various machine learning techniques. The dataset used is from a popular machine learning competition hosted on Kaggle.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Predictive System](#predictive-system)
- [Usage](#usage)
- [License](#license)

## Introduction

The goal of this project is to predict the sales of products in different stores of a big mart chain based on various attributes of the products and stores. This can help the management to understand the factors affecting sales and strategize accordingly.

## Dataset

The dataset used in this project is the "Big Mart Sales" dataset, which includes information about various products across different outlets. The dataset can be found on Kaggle [here](https://www.kaggle.com/brijbhushannanda1979/bigmart-sales-data).

### Categorical Features

- Item_Identifier
- Item_Fat_Content
- Item_Type
- Outlet_Identifier
- Outlet_Size
- Outlet_Location_Type
- Outlet_Type

### Numerical Features

- Item_Weight
- Item_Visibility
- Item_MRP
- Item_Outlet_Sales

## Dependencies

To run this project, you need the following dependencies:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.impute import SimpleImputer
import joblib
```

## Data Preprocessing

The following steps were performed to preprocess the data:

1. Handling Missing Values:
    - Replaced missing values in the `Item_Weight` column with the mean.
    - Replaced missing values in the `Outlet_Size` column with the mode.

2. Encoding Categorical Variables:
    - Used `LabelEncoder` to convert categorical features into numerical values.

## Exploratory Data Analysis (EDA)

Performed EDA to understand the distribution and relationships of various features. This includes plotting distributions and counts for numerical and categorical features.

```python
# Example of distribution plot
plt.figure(figsize=(4,4))
sns.distplot(big_mart_data['Item_Weight'])
plt.show()
```

## Model Training

Used XGBoost Regressor to train the model:

```python
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)
```

## Evaluation

Evaluated the model performance using R-squared metric:

```python
# Prediction on training data
training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)

# Prediction on test data
test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
```

## Predictive System

Implemented a predictive system to make predictions on new data:

```python
def predictive_system(input_data):
    # Preprocessing input data and making predictions
    prediction = regressor.predict(input_data)
    return prediction[0]
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/big-mart-sales-prediction.git
```

2. Navigate to the project directory:

```bash
cd big-mart-sales-prediction
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the script to train the model and make predictions.

5. Example usage of the predictive system:

```python
example_input = {
    'Item_Identifier': 'FDA15',
    'Item_Weight': 9.3,
    'Item_Fat_Content': 'Low Fat',
    'Item_Visibility': 0.016047301,
    'Item_Type': 'Dairy',
    'Item_MRP': 249.8092,
    'Outlet_Identifier': 'OUT049',
    'Outlet_Establishment_Year': 1999,
    'Outlet_Size': 'Medium',
    'Outlet_Location_Type': 'Tier 1',
    'Outlet_Type': 'Supermarket Type1'
}

predicted_sales = predictive_system(example_input)
print("Predicted Item Outlet Sales:", predicted_sales)
```

