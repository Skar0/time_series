## Time Series Forecasting Using Neural Networks and Statistical Models

The goal of this project is to forecast future web traffic for Wikipedia articles using different techniques ranging from statistical models to deep neural networks. A description of the project, along with examples of our predictions is provided below.

<img src="https://github.com/Skar0/time_series/blob/master/training_visualization/prediction_comparison_0.png">

## Achievement
This project was created as part of a Kaggle competition. Our best submission ranked 3rd out of 32 teams entered in the competition.

## Data
### Dataset

The dataset provided for the competition consists in 45 time series created from the original 145k timeseries of daily page       views on Wikipedia articles from the Web Traffic Time Series Forecasting competition (https://www.kaggle.com/c/web-traffic-time-series-forecasting) on Kaggle. Each series starts from July, 1st, 2015 and ends on August, 20th, 2017. The task at hand is to forecast the number of daily views for the 21-day period starting from August, 21st, 2017 up until September 10th, 2017. The metric used to measure the accuracy of our predictions is the Symmetric Mean Absolute Percentage Error (SMAPE).

### Data preparation

In order to train models to perform forecasting, we first needed to prepare the data. Before creating samples from the time series to feed our models, we first performed outliers removal and data normalization. Outliers removal was performed using two methods: the quantile method and the modified z-score method. Normalization was performed using min-max scaling. The last step of data preparation was to split the dataset into train and validation sets.

## Project structure
Our goal was to create a project which approaches the Kaggle competition using a series of increasingly more complex models. We tried to make the functions as legible as possible to improve readability of our code. The names of the modules and functions are quite self-explanatory. The code is organized as follows:

```bash
├── arima_models
│   ├── __init__.py
│   └── arima_forecasting.py
├── neural_networks_models
│   ├── __init__.py
│   ├── LSTM.py
│   ├── lstm_multi_step_forecasting.py
│   ├── nn_multi_step_forecasting.py
│   └── nn_single_step_forecasting.py
├── regression_models
│   ├── __init__.py
│   ├── regression_forecasting.py
│   ├── regression_preprocessing.py
│   └── regression_tools.py
├── notebooks
│   ├── SARIMAX.ipynb
│   ├── good_regression_forecasting.ipynb
│   └── xgboost_plus_weights.ipynb
├── submissions
│   ├── sarima_scaled.csv
│   ├── sarima_scaled_exog_all.csv
│   ├── sarima_scaled_exog.csv
│   ├── submission_arima.csv
│   ├── submission_arima_outliers2.csv
│   ├── submission_arima_outliers.csv
│   └── submission_prophet.csv
├── data
│   └── train.csv
├── training_visualization
│   ├── series-*-comparison.pdf
│   └── series-*-submission.pdf
├── find_submission_models.py
├── utils.py
├── README.md
└── .gitattributes
```
## The models

We evaluated three types of models. First, ARIMA models were tested with and without exogenous data. We estimated ARIMA parameters using the auto arima function from pmdarima. Then, we used regression-based techniques such as linear regression using scikit-learn and gradient boosting using XGBoost. Finaly, we attempted to perform forecasting with neural network models using Keras. These models have different architectures which range from simple multi-layer perceptrons to LSTMs. The find_submission_models module performs model fitting and evaluation on each series and selects the best models for submission.

## Requirements:
The project was tested with the following versions of libraries:

      Keras==2.2.4
      matplotlib==2.2.3
      numpy==1.18.1
      pandas==0.23.4
      pmdarima==1.5.2
      scikit-learn==0.19.2
      scipy==1.4.1
      statsmodels==0.11.0
      tensorflow==1.11.0
      xgboost==0.81
   
## Results

Below are a few examples showing the peformance of our models:

<img src="https://github.com/Skar0/time_series/blob/master/training_visualization/series-9.png" width=400>
<img src="https://github.com/Skar0/time_series/blob/master/training_visualization/prediction_comparison_5.png">
<img src="https://github.com/Skar0/time_series/blob/master/training_visualization/validation_comparison_5.png">
<img src="https://github.com/Skar0/time_series/blob/master/training_visualization/prediction_comparison_4.png">
<img src="https://github.com/Skar0/time_series/blob/master/training_visualization/validation_comparison_4.png">
<img src="https://github.com/Skar0/time_series/blob/master/training_visualization/prediction_comparison_3.png">
<img src="https://github.com/Skar0/time_series/blob/master/training_visualization/validation_comparison_3.png">
<img src="https://github.com/Skar0/time_series/blob/master/training_visualization/prediction_comparison_2.png">
<img src="https://github.com/Skar0/time_series/blob/master/training_visualization/validation_comparison_2.png">
