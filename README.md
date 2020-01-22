# Time series forecasting using neural networks and statistical models

This project is a collection of models and techniques used to predict time series.


A description of the project, along with examples of our predictions is provided below.

## Achievement
This project was created as part of a Kaggle competition. Our submission ranked 3rd out of 32 teams entered in the competition.

## Datasets
#### Provided dataset

The dataset consists in 45 time series created from teh Web Traffic Forecasting competition. It is a combination of different 
series and therefore the models used for that competition would not be the best for this competition. The series consisted of 
data from  2015-07-01 to 2017-08-20 and the task was to forecast the values for the next 21 days.

#### Data preparation

In order to fit models, we first preprocessed data by removing outliers and normalizing it. To do so, we used the quantile method and
the method.

## Project structure
Our goal was to create a project which handles the competition as a series of increasingly complex models. We tried to make the 
functions as legible as possible. We also made sure to create functions that could be launched on remote machines for training and still
give us visualisation of our progress. The code is organized as follows :

```bash
├── arima_models
│   ├── arima_forecasting.py
│   └── __init__.py
├── data
│   └── train.csv
├── find_submission_models.py
├── neural_networks_models
│   ├── __init__.py
│   ├── lstm_multi_step_forecasting.py
│   ├── nn_multi_step_forecasting.py
│   └── nn_single_step_forecasting.py
├── notebooks
│   ├── good_regression_forecasting.ipynb
│   └── xgboost_plus_weights.ipynb
├── one_step_regression.py
├── README.md
├── regression_models
│   ├── __init__.py
│   ├── regression_forecasting.py
│   ├── regression_preprocessing.py
│   └── regression_tools.py
├── submissions
│   ├── sarima_scaled.csv
│   ├── sarima_scaled_exog_all.csv
│   ├── sarima_scaled_exog.csv
│   ├── submission_arima.csv
│   ├── submission_arima_outliers2.csv
│   ├── submission_arima_outliers.csv
│   └── submission_prophet.csv
├── tools.py
├── utils.py
└── xgboost_plus_weights.ipynb
```

For this competition, we evaluated three types of models. Each model was tested on a validation series provided while fitting.
First, ARIMA-based models were tested. We estimated ARIMA parameters using auto arima from pmdarima. Then we used regression-based techniques such as linear regression and gradientboosting using xgboost.
Lastly, we tried neural network models. First by predicting several steps at the time and then one step at the time. Finaly, LSTM were tried.
find_submission_models performed model fitting and evaluation on each series and selected best models for submission.

## How to use

#### Requirements:
The project was tested with the following versions of librairies:

TODO
   
#### Launcher

TODO

## Results

#### Performance
Our submission got the third place in the competition. Here are a few examples of our model.
