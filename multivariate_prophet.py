from prophet import Prophet
import pandas as pd
from datetime import datetime
import os
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('multivariate_prophet')

#This class will input a dataframe which has been formatted with 'time' (ds in prophet speak) 'primary' column (y in prophet speak) and our regressors which can be anything.
#The class will then create a prophet model for the primary column and each regressor column.
#The class will then store the models in a dictionary for easy access.
#The class will also have a method to predict the future values of each regressor column independantly to prepare a future dataframe for prediction of the primary column.
#variables you can input stuff like holidays, seasonality, etc. IE {'daily_seasonality': 10, 'interval_width': 0.80}

class MultivariateProphet():
   def __init__(self) -> None:
      
      #Create a dictionary to hold prophet models for each regressor
      self.model = {}
      self.regressors = []
      self.primary = None 
      self.forecast_df = None

   def train_model(self, dataframe: pd.DataFrame, variables: dict = None, add_seasonality: list[dict] = None, primary: str = "y", time: str = "ds", floor: str ="floor", cap: str ="cap", standardize_regressors: bool=True ) -> None:
      '''
      Use this function to train a prophet model on a dataframe with multiple regressors and predict those regressors. The model is stored in the self.model[primary] dictionary. We also store the models for each regressor in the self.model dictionary as well for debugging purposes.

      :param dataframe: pd.DataFrame - The dataframe containing the data to train the model on. Must contain columns for 'time' (ds in prophet speak), 'primary' (y in prophet speak), and any regressors you want to use. Note by default it will add regressors as additive. If you want to add them as multiplicative you must add 'multiplicative' to the column name. IE 'regressor_multiplicative' and it will be treated as a multiplicative regressor.
      :param variables: dict - A dictionary of variables to pass to the prophet model. Leave this blank to use the default training criteria IE {'daily_seasonality': 10, 'interval_width': 0.80}
      :param add_seasonality: dict - A dictionary of custom seasonality to add to the model. The key is the name of the seasonality and the value is a dictionary with the period and fourier_order. IE {"name": "monthly", "period": 30.5, "fourier_order": 5}}
      :param primary: str - The name of the primary column in the dataframe. This will be the column we are predicting and will be renamed to y in the prophet model. Predictions will become y_hat
      :param time: str - The name of the time column in the dataframe. This will be renamed to ds in the prophet model.
      :param floor: str - The name of the floor column in the dataframe. This will be used to set a floor for the model. Leave it blank to not use a floor. (IE If your dataset is distance traveled then you want to set the floor to 0)
      :param cap: str - The name of the cap column in the dataframe. This will be used to set a cap for the model. Leave it blank to not use a cap. (IE if your column is a percentage, you may want to cap it at 100)
      :param standardize_regressors: bool - Set to True to standardize the regressors. Note (the fbprophet library will standardize the regressors by default.) Standardization is a preprocessing step that transforms the data to have a mean of 0 and a standard deviation of 1.

      :return: None (The model is stored in the self.model[primary] dictionary. We also store the models for each regressor in the self.model dictionary as well for debugging purposes.)
      '''   
      start_time = datetime.now()
      if primary not in dataframe.columns:
            raise ValueError(f"Primary column {primary} not found in dataframe.")
      
      self.primary = primary

      if time not in dataframe.columns:
            raise ValueError(f"Time column {time} not found in dataframe.")
      
      self.regressors = [col for col in dataframe.columns if col not in [primary, time, floor, cap]]
      
      #Add the primary model to our self.model dictionary
      self.model[primary] = Prophet(**variables)
   
      try:
         if len(self.regressors) > 0:
            for regressor in self.regressors:
               #create a dataframe for each regressor with timeframe and regressor as y (from columns {time} and {regressor} in the dataframe)
               regressor_df = dataframe[[time, regressor]].rename(columns={time: 'ds', regressor: 'y'})
               #create a prophet model for each regressor unpacking (**) the variables dictionary
               self.model[regressor] = Prophet(**variables)
               #Add custom seasonality if provided to regressor models
               if add_seasonality:
                   for seasonality_params in add_seasonality:
                       self.model[regressor].add_seasonality(**seasonality_params)
               #train each regressor model
               self.model[regressor].fit(regressor_df)
               #add each regressor to the primary model
               if 'multiplicative' in regressor:
                  self.model[primary].add_regressor(regressor, mode='multiplicative', standardize=standardize_regressors)
               else:
                  self.model[primary].add_regressor(regressor, standardize=standardize_regressors)
               
      
      except Exception as e:
         raise ValueError(f"Error training model for regressor {regressor}: {e}")
      
      #In our primary dataframe rename the time column to ds an primary to y
      dataframe = dataframe.rename(columns={time: 'ds', primary: 'y'})
      if floor:
         dataframe['floor'] = floor
      if cap:
         dataframe['cap'] = cap

      if add_seasonality:
                   for seasonality_params in add_seasonality:
                       self.model[primary].add_seasonality(**seasonality_params)
      #train the primary model (the dataframe already contains all of our regressor data)
      self.model[primary].fit(dataframe)
      #store the training date
      self.model["training_date"] = datetime.now()
      end_time = datetime.now()
      print(f"Model with {len(dataframe)} datapoints trained in {end_time - start_time} seconds.")
         
   def forecast(self, periods: int, floor=None, cap=None, freq='min') -> pd.DataFrame:
      '''
      Use this function to forecast future values of the primary column. This function will also predict the future values of each regressor column independantly to prepare a future dataframe for prediction of the primary column.
      :param periods: int - The number of periods to forecast into the future
      :param floor: int - The floor value for the primary column. Leave it blank to not use a floor. (IE If your dataset is distance traveled then you want to set the floor to 0)
      :param cap: int - The cap value for the primary column. Leave it blank to not use a cap. (IE if your column is a percentage, you may want to cap it at 100)
      :param freq: str - The frequency of the forecast. Default is 'min' for minutes. You can use 'H' for hours, 'D' for days, 'W' for weeks, 'M' for months, 'Y' for years.

      :return pd.DataFrame - A dataframe containing the forecasted values of the primary column.
      '''
      if not self.model:
         raise ValueError("Model has not been trained or loaded. Train or load a model first.")
   
      future_df = self.model[self.primary].make_future_dataframe(periods=periods, freq=freq, include_history=False)
      if floor:
         future_df['floor'] = floor
      if cap:
         future_df['cap'] = cap
      #Run through each regressor and predict the future values
      if len(self.regressors) > 0:
         for regressor in self.regressors:
            regressor_future_df = self.model[regressor].make_future_dataframe(periods=periods, freq=freq, include_history=False)
            logger.debug(f"Future dataframe for regressor {regressor}: {regressor_future_df.shape}")
            
            regressor_forecast = self.model[regressor].predict(regressor_future_df)
            logger.debug(f"Forecast for regressor {regressor}: {regressor_forecast.shape}")
            
            future_df[regressor] = regressor_forecast['yhat']
            
            # Just in case we end up predicting any NaN values
            if future_df[regressor].isnull().any():
                logger.debug(f"There are {future_df[regressor].isnull().count()} null values. Imputing missing values for regressor {regressor}")
                # Simple imputation with median; you can choose a method appropriate for your data
                future_df[regressor].fillna(future_df[regressor].median(), inplace=True)
      
      #Save our future dataframe for later use a debugging
      self.forecast_df = future_df.copy()
      forecast = self.model[self.primary].predict(future_df)
      logger.debug(f"Forecast for primary column {self.primary}: {forecast.shape}")
      return forecast
   