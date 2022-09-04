import traceback
from typing import Any
import json
import joblib

import pandas as pd
import numpy as np
import importlib

from logger import logging
logger = logging.getLogger(__name__)

def create_date_features(df: pd.DataFrame, date_feature_column: str = 'date', drop_features: list = []) -> pd.DataFrame:
    ''' This function converts the "date" column into pandas datetime.\n
        This function also creates new features with year, month, and day.

        Parameters:
        - df: pd.DataFrame
        - date_feature_column: str
            it is a date column, this column is used to create new date features

        Returns:
        - pd.DataFrame
    '''
    df[date_feature_column] = pd.to_datetime(df[date_feature_column])
    df['year'] = df[date_feature_column].dt.year
    df['month'] = df[date_feature_column].dt.month
    df['day'] = df[date_feature_column].dt.day

    df['year'] = df['year'].astype(dtype='float64')
    df['month'] = df['month'].astype(dtype='float64')
    df['day'] = df['day'].astype(dtype='float64')

    df.drop(
        labels=[date_feature_column],
        axis=1,
        inplace=True
    )

    if len(drop_features):
        df.drop(
        labels=drop_features,
        axis=1,
        inplace=True
    )

    return df

def clean_dataframe(df: pd.DataFrame, target: str = '') -> pd.DataFrame:
    ''' This function will perform the following functions.
        (1) drop the features that have more than 50% missing values
        (2) filling NONE in categorical features
        (3) filling median in numerical features
        In both the categorical and numerical features case, it will create a new column with 1 at missing value otherwise 1

        Parameters:
        - df: pd.DataFrame
        - target: str
            target column name in the dataframe, only use this for dataframe with target column

        Returns:
        - pd.DataFrame
    '''
    target = [target]

    high_missing_value_columns = []
    total_rows = len(df)
    for feature in df:
        nans = df[feature].isna().sum()
        if (100*nans) / total_rows > 50.0:
            high_missing_value_columns.append(feature)
    
    df.drop(
        labels=high_missing_value_columns,
        axis=1,
        inplace=True
    )

    categorical_features = []
    for column in df.columns:
        if df[column].dtype == 'object' and column not in target:
            categorical_features.append(column)
    for cf in categorical_features:
        df[f'{cf}_na'] = np.where(
            df[cf].isna(),
            1,
            0
        )
        df[cf].fillna(
            value='NONE',
            inplace=True
        )

    numerical_features = []
    for column in df.columns:
        if column not in categorical_features and column not in target:
            numerical_features.append(column)
    for nf in numerical_features:
        if df[nf].isna().sum() > 1:
            median_value = df[nf].median()
            df[f'{nf}_na'] = np.where(
                df[nf].isna(),
                1,
                0
            )
            df[nf].fillna(
                value=median_value,
                inplace=True
            )

    return df

def get_class_reference(module_name: str, class_name: str) -> Any:
    ''' Get the class reference to generate model dynamically

        Parameters:
        - module_name: str
            this is the name of the module, example: sklearn.linear_model
        - class_name: str
            this is the name of the class, example: LinearRegressor

        Returns:
        - Any
    '''
    try:
        module = importlib.import_module(module_name)
        class_reference = getattr(module, class_name)
        return class_reference
    except:
        logger.exception(f'Uncaught exception - {traceback.format_exc()}')

def update_property_of_class(instance_reference: object, parameters: dict) -> Any:
    ''' Update the class attributes to generate model dynamically

        Parameters:
        - instance_reference: object
            this is the instance of the model
        - parameters: dict
            this is the dictionary of the parameters to set in the model instance

        Returns:
        - Any
    '''
    try:
        if not isinstance(parameters, dict):
            raise Exception('property_data parameter must be a dictionary.')
        for key, value in parameters.items():
            setattr(instance_reference, key, value)
        return instance_reference
    except:
        logger.exception(f'Uncaught exception - {traceback.format_exc()}')

def read_json_file(file_path: str) -> dict:
    ''' This function will read a json file.

        Parameters:
        - file_path: str

        Returns:
        - Python dict
    '''
    try:
        with open(file_path, 'r') as file:
            model_data = json.load(file)
            return model_data
    except:
        logger.exception(f'Uncaught exception - {traceback.format_exc()}')

def write_json_file(dictionary: dict, file_path: str) -> None:
    ''' This function will write a json file.

        Parameters:
        - dictionary: dict
        - file_path: str

        Returns:
        - None
    '''
    try:
        json_object = json.dumps(dictionary, indent=2)
        with open(file_path, 'w') as file:
            file.write(json_object)
    except:
        logger.exception(f'Uncaught exception - {traceback.format_exc()}')

def save_object(object: object, file_path: str) -> None:
    ''' This function will save an object.

        Parameters:
        - object: object
        - file_path: str

        Returns:
        - None
    '''
    try:
        with open(file_path, 'wb') as file:
            joblib.dump(
                value=object,
                filename=file
            )
    except:
        logger.exception(f'Uncaught exception - {traceback.format_exc()}')

def load_object(file_path: str) -> object:
    ''' This function will load an object.

        Parameters:
        - file_path: str

        Returns:
        - object
    '''
    try:
        with open(file_path, 'rb') as file:
            model_object = joblib.load(filename=file)
            return model_object
    except:
        logger.exception(f'Uncaught exception - {traceback.format_exc()}')

def save_numpy_array(array: np.ndarray, file_path: str) -> None:
    ''' This function will save a numpy array.

        Parameters:
        - array: np.ndarray
        - file_path: str

        Returns:
        - None
    '''
    try:
        with open(file_path, 'wb') as file:
            np.save(
                file=file,
                arr=array
            )
    except:
        logger.exception(f'Uncaught exception - {traceback.format_exc()}')

def load_numpy_array(file_path: str) -> np.ndarray:
    ''' This function will load a numpy array.

        Parameters:
        - file_path: str

        Returns:
        - np.ndarray
    '''
    try:
        with open(file_path, 'rb') as file:
            array = np.load(file=file)
            return array
    except:
        logger.exception(f'Uncaught exception - {traceback.format_exc()}')

def compute_weight(score: list) -> list:
    ''' Compute the weight/probabilty of the score.

        Parameters:
        - score: str

        Returns:
        - list
    '''
    score = np.subtract(0, score)
    e_x = np.exp(score - np.max(score))
    return e_x / e_x.sum()
