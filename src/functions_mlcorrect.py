# =============================================================================
#
# This scripts contains functions used by mlcorrect_*.py for both corr and SO
#
# Copyright (C) 2021-2024 Julius Kleine Büning
#
# =============================================================================


import os
import numpy as np
import pandas as pd

# Suppress info and warning messages from tensorflow
# (Print: 0: INFO, WARNING, ERROR; 1: WARNING, ERROR; 2: ERROR; 3: none)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

########## GLOBAL DECLARATIONS ##########

# define the version of mlcorrect_*.py (this does not (yet) exist for getdata_*.py)
version_mlcorrect = '1.0.0'

########## END GLOBAL DECLARATIONS ##########



def max_error_neg(y_true, y_pred) -> float:
    """Create custom metric: maximum negative error (MAX_neg).
    
    ATTENTION: The value can be positive (if all error are positive).
               This metric gives the minimum (signed) error value.
    
    Args:
        y_true: Target values in a 2-dimensional set of data (tf.Tensor object).
        y_pred: Predicted values in a 2-dimentional set of data (tf.Tensor object).

    Returns:
        Minimum error value (predicted - target).
    """

    errors = y_pred - y_true
    return tf.reduce_min(errors)


def max_error_pos(y_true, y_pred) -> float:
    """Create custom metric: maximum positive error (MAX_pos).
    
    ATTENTION: The value can be negative (if all error are negative).
               This metric gives the maximum (signed) error value.
    
    Args:
        y_true: Target values in a 2-dimensional set of data (tf.Tensor object).
        y_pred: Predicted values in a 2-dimentional set of data (tf.Tensor object).

    Returns:
        Maximum error value (predicted - target).
    """

    errors = y_pred - y_true
    return tf.reduce_max(errors)


def mean_error(y_true, y_pred) -> float:
    """Create custom metric: mean (signed) error (ME).
    
    Args:
        y_true: Target values in a 2-dimensional set of data (tf.Tensor object).
        y_pred: Predicted values in a 2-dimensional set of data (tf.Tensor object).

    Returns:
        Mean error (predicted - target).
    """

    errors = y_pred - y_true
    return tf.reduce_mean(errors, axis=-1)


def rmse(y_true, y_pred) -> float:
    """Create custom metric: root mean squared error (RMSE, or deviation, RMSD).
    
    Args:
        y_true: Target values in a 2-dimensional set of data (tf.Tensor object).
        y_pred: Predicted values in a 2-dimensional set of data (tf.Tensor object).

    Returns:
        Root mean square error.
    """

    errors = y_pred - y_true               # errors
    squared_errors = errors ** 2           # squared errors
    mse = tf.reduce_mean(squared_errors)   # mean squared error
    rmse = tf.sqrt(mse)                    # root mean squared error
    return rmse


def standard_deviation(y_true, y_pred) -> float:
    """Create custom metric: standard deviation (SD).
    
    Args:
        y_true: Target values in a 2-dimensional set of data (tf.Tensor object).
        y_pred: Predicted values in a 2-dimensional set of data (tf.Tensor object).

    Returns:
        Standard deviation of the errors.
    """

    errors = y_pred - y_true
    return tf.math.reduce_std(errors)


def all_equal(stats_list) -> bool:
    """Check if all elements in a 1-dimensional pd.DataFrame array are the same.

    This may be used for security when trying to norm values in the list.
    
    Args:
        stats_list: List to check (pd.DataFrame object).

    Returns:
        True if all elements are equal, else False.
    """

    alleq = True
    first_key = stats_list.keys()[0]      # get arbitrary key of the list
    first_value = stats_list[first_key]   # get an arbitrary value of the list
    for value in stats_list:
        if value != first_value:
            alleq = False
            break

    return alleq


def linear_fit(data_low: pd.Series, data_target: pd.Series):
    """Perform linear regression on some input data via numpy.polyfit().

    Args:
        data_low: Array with low-level DFT data of the data set (pd.Series object).
        data_target: Array with target data (high-level - low-level) of the data set (pd.Series object).
    
    Returns:
        Slope of the linear function.
        Intercept of the linear function.
        Coefficient of determination (Pearson R^2) of the linar fit.
    """

    xtrain = (data_target + data_low).to_numpy()
    ytrain = data_low.to_numpy()
    slope, intercept = np.polyfit(xtrain, ytrain, 1)
    r_squared = np.corrcoef(xtrain, ytrain)[0,1]**2

    return slope, intercept, r_squared


def scale_linear(data: pd.Series, slope: float, intercept: float) -> pd.Series:
    """Perform linear regression correction on some data with given slope and intercept.

    Args:
        data: Array with the original input data (pd.Series object).
        slope: Slope of linear fit function obtained by linear_fit().
        interecpt: intercept of linear fit function obtained by linear_fit().
    
    Returns:
        Array with the scaled data (pd.Series object).
    """

    return (data - intercept)/slope


def norm(data: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Normalize the trainig set data.
    
    Data on different scales and ranges (not normed) makes training more difficult and is input-unit-dependent.

    Args:
        data: The whole data set to be normed (pd.DataFrame object).
        stats: Statisic measures of the data obtained from data.describe() (pd.DataFrame object).

    ATTENTION: stats must be set up such that
        The same value for all entries in a column will lead to NaN values.
        Make sure stats is set up such that this case is avoided.

    Returns:
        The normed variant of data (pd.DataFrame object).
    """

    return (data - stats['mean']) / stats['std']

