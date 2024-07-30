# multivariate_prophet

## Overview
This class allows you to use Facebook prophet for multivariate forecasting (each variable is independently forecasted, and than the primary forecast is generated). It works the same as facebooks prophet model so if you are familiar with that you will be able to use this class.


MultivariateProphet is an implementation of Facebook's Prophet forecasting tool, designed to handle multiple regressors in time series forecasting, automatically prepare their dataframs and output a prediction dataframe for your primary series. This class extends the capabilities of the original Prophet model by allowing for the incorporation and prediction of multiple regressor variables alongside the primary time series without having to pre-populate future regressor data.

## Features

- Multivariate forecasting with support for multiple regressors
- Automatic training of individual Prophet models for each regressor
- Flexible configuration options for model parameters and seasonality
- Support for both additive and multiplicative regressors
- Built-in logging for debugging and tracking model performance
- Standardization option for regressors

## Requirements

- Python 3.x
- pandas
- fbprophet
- numpy
- logging

## Installation

1. Ensure you have Python 3.x installed on your system.
2. Install the required packages:

#bash
pip install pandas fbprophet numpy

Download the multivariate_prophet.py file and place it in your project directory.

Usage
Importing the Class
from multivariate_prophet import MultivariateProphet

Creating an Instance
model = MultivariateProphet()

Training the Model
model.train_model(
    dataframe=your_dataframe,
    variables={'daily_seasonality': 10, 'interval_width': 0.80},
    add_seasonality=[{"name": "monthly", "period": 30.5, "fourier_order": 5}],
    primary="target_column",
    time="date_column",
    floor="min_value_column",
    cap="max_value_column",
    standardize_regressors=True
)

forecasting
forecast = model.forecast(periods=30, floor=0, cap=100, freq='D')

API Reference
MultivariateProphet Class
__init__(self)
Initializes the MultivariateProphet instance.
train_model(self, dataframe: pd.DataFrame, variables: dict = None, add_seasonality: list[dict] = None, primary: str = "y", time: str = "ds", floor: str = "floor", cap: str = "cap", standardize_regressors: bool = True) -> None
Trains the Prophet model on the provided dataframe with multiple regressors.
Parameters:

dataframe: pd.DataFrame - The input dataframe containing the time series data and regressors.
variables: dict - (Optional) A dictionary of variables to pass to the Prophet model.
add_seasonality: list[dict] - (Optional) A list of dictionaries specifying custom seasonality parameters.
primary: str - The name of the primary target column in the dataframe.
time: str - The name of the time column in the dataframe.
floor: str - (Optional) The name of the floor column in the dataframe.
cap: str - (Optional) The name of the cap column in the dataframe.
standardize_regressors: bool - Whether to standardize the regressors (default: True).

forecast(self, periods: int, floor=None, cap=None, freq='min') -> pd.DataFrame
Generates forecasts for the primary column and all regressors.
Parameters:

periods: int - The number of periods to forecast into the future.
floor: int - (Optional) The minimum value for the forecast.
cap: int - (Optional) The maximum value for the forecast.
freq: str - The frequency of the forecast (default: 'min' for minutes).

Returns:

pd.DataFrame - A dataframe containing the forecasted values.

Best Practices

Ensure your input dataframe is properly formatted with a time column, primary target column, and any desired regressor columns.
Use meaningful column names for your time, primary, and regressor variables.
Consider the nature of your regressors (additive or multiplicative) when naming them in the dataframe.
Utilize the variables parameter to fine-tune the Prophet model for your specific use case.
Add custom seasonality when appropriate for your time series data.
Use the standardize_regressors option judiciously based on the nature of your regressor variables.
Always check the returned forecast dataframe for any anomalies or unexpected results.

Troubleshooting

If you encounter issues with missing values in regressor forecasts, the model will attempt to impute them using the median value. Check the logs for any warnings about this process.
Ensure that your input dataframe does not contain any NaN or infinite values, as these can cause issues during model training.
If you experience performance issues with large datasets, consider reducing the number of regressors or the amount of historical data used for training.

Contributing
Contributions to improve MultivariateProphet are welcome. Please feel free to submit pull requests or open issues on the project repository.

License
MIT LISCENCE

Acknowledgments
This project is built upon Facebook's Prophet library. Thanks guys for your contribution to the field of time series forecasting.