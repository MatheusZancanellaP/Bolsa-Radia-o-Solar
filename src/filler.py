import pandas as pd
import numpy as np
from pydmd import DMD
from statsmodels.tsa.arima.model import ARIMA

def fill_df(dataframe, filler):
    data = {}
    for column in dataframe.columns:
        data[column] = filler(dataframe[column])
    return pd.DataFrame(data=data, index=dataframe.index.values)


def interpolate_filler(data):
    return data.interpolate(method='linear', limit_direction='backward')


def dmd_filler(data: pd.Series):
    column = data.name
    data = data.to_frame()
    filled_data = data.copy()
    matrix_data = filled_data.copy().values.T

    col_mean = np.nanmean(matrix_data, axis=1)
    inds_nan = np.where(np.isnan(matrix_data))
    matrix_data[inds_nan] = np.take(col_mean, inds_nan[0])

    dmd = DMD(svd_rank=matrix_data.shape[0])
    dmd.fit(matrix_data)
    dmd_data = dmd.reconstructed_data.real.T
    filled_data.values[np.isnan(data.values)] = dmd_data[np.isnan(data.values)]

    return filled_data[column]


def arima_filler(data, order=(2,0,3)):
    # Fills NaN values using ARIMA model predictions.
    #
    # Parameters:
    #     data (pd.Series): The time series data with NaN values to fill.
    #     order (tuple): The (p,d,q) order of the ARIMA model.
    #
    # Returns:
    #     pd.Series: The input time series with NaN values filled using ARIMA.
    
    filled_data = data.copy()
    
    # Find the NaN values
    nan_inds = np.isnan(data)
    
    # If there are NaN values, fill them
    if np.any(nan_inds):
        # Fit ARIMA model on non-NaN values
        model = ARIMA(data[~nan_inds], order=order).fit()
        
        # Predict the NaN values
        predictions = model.predict(start=data.index.get_loc(data[nan_inds].index[0]), 
                                    end=data.index.get_loc(data[nan_inds].index[-1]))
        
        # Replace NaN values with predictions
        filled_data[nan_inds] = predictions.values
    
    return filled_data