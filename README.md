# Stock-Prediction-Models

This repository is created to store different trained model and their results of executions in a structured manner accoring to their timeline and value.

1. To name a .ipynb file: **5_15_23_open.h5**

```text
("n-Year"_ "StartDate" _ "End Date" _ "feature name"."file extension")
```

2. To name a model file: **5Model_15_23.h5**

```text
("n-Year"_ "StartDate" _ "End Date" _ "feature name"."file extension")
```


## Model Details and Navigation (Without XGBoost)

1. For Close Feature

| 5 year Model | 8 Year Model | 10 Year Model |
|--------------|:-----:|-----------:|
|<a href="/5 year Model/Close Feature/75/5Model_18_23_75.ipynb">Epoch 75 </a>   | <a href="/8 year Model/Close Feature/75/8model_15_23.ipynb">Epoch 75 </a>      |  <a href="/10 year Model/Close Feature/75/10Model_13_23_75.ipynb">Epoch 75 </a>  |
|<a href="/5 year Model/Close Feature/100/5Model_18_23_100.ipynb">Epoch 100 </a>|  <a href="/8 year Model/Close Feature/100/8model_15_23_100.ipynb">Epoch 100 </a> | <a href="/10 year Model/Close Feature/100/10Model_13_23_100.ipynb">Epoch 100 </a>   |
|<a href="/5 year Model/Close Feature/125/5Model_18_23_125.ipynb">Epoch 125 </a> |   <a href="/8 year Model/Close Feature/125/8model_15_23_125.ipynb">Epoch 125 </a>    |    <a href="/10 year Model/Close Feature/125/10Model_13_23_125.ipynb">Epoch 125 </a>       |
|<a href="/5 year Model/Close Feature/150/5Model_18_23_150.ipynb">Epoch 150 </a> |  <a href="/8 year Model/Close Feature/150/8model_15_23_150.ipynb">Epoch 150 </a>     |    <a href="/10 year Model/Close Feature/150/10Model_13_23_150.ipynb">Epoch 150 </a>       |

2. For Open Feature

| 5 year Model | 8 Year Model | 10 Year Model |
|--------------|:-----:|-----------:|
|<a href="/5 year Model/Open Feature/75/5Model_18_23_75_open.ipynb">Epoch 75 </a>   | <a href="/8 year Model/Open Feature/75/8Model_15_23_75_open.ipynb">Epoch 75 </a>      |  <a href="/10 year Model/Open Feature/75/10Model_13_23_75_open.ipynb">Epoch 75 </a>  |
|<a href="/5 year Model/Open Feature/100/5Model_18_23_100_open.ipynb">Epoch 100 </a>|  <a href="/8 year Model/Open Feature/100/8Model_15_23_100_open.ipynb">Epoch 100 </a> | <a href="/10 year Model/Open Feature/100/10Model_13_23_100_open.ipynb">Epoch 100 </a>   |
|<a href="/5 year Model/Open Feature/125/5Model_18_23_125_open.ipynb">Epoch 125 </a> |   <a href="/8 year Model/Open Feature/125/8Model_15_23_125_open.ipynb">Epoch 125 </a>    |    <a href="/10 year Model/Open Feature/125/10Model_13_23_125_open.ipynb">Epoch 125 </a>       |
|<a href="/5 year Model/Open Feature/150/5Model_18_23_150_open.ipynb">Epoch 150 </a> |  <a href="/8 year Model/Open Feature/150/8Model_15_23_150_open.ipynb">Epoch 150 </a>     |    <a href="/10 year Model/Open Feature/150/10Model_13_23_150_open.ipynb">Epoch 150 </a> |

XG Boost can be used as a standalone model, its strength lies in its ability to combine multiple weak models (decision trees) into a stronger ensemble. XGBoost uses a gradient boosting framework, where each subsequent tree is trained to correct the errors of the previous trees, leading to improved overall performance.

## Grid Search CV

Grid Search CV is a popular hyperparameter tuning technique used in machine learning to find the optimal combination of hyperparameters for a given model. In XGBoost, a gradient boosting algorithm, Grid Search CV is particularly valuable due to the numerous hyperparameters that can significantly impact the model's performance.   

```py
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5)

# Fit the grid search to the data
grid_search.fit(lstm_features_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use the best model to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(lstm_features_test)
```
