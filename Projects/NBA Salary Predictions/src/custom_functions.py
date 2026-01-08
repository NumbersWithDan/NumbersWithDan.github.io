import os
from pathlib import Path
from kaggle import KaggleApi
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


def download_kaggle_dataset(dataset, path, unzip = True):
    '''
    Authenticates using the Kaggle API and downloads a dataset to the specified path.

    Parameters
    ----------
    dataset : str
        Kaggle dataset identifier in the form 'owner/dataset-name'.
    path : str
        Local directory where the dataset should be saved.
    unzip : bool, optional (default=True)
        Whether to automatically unzip the dataset after downloading.

    Returns
    -------
    None
    '''
    api = KaggleApi()
    api.authenticate()

    # Download the dataset to the specified path
    api.dataset_download_files(dataset, path=path, unzip=unzip)
    print(f"Dataset '{dataset}' downloaded to '{path}'")

    return None


def impute_and_scale(player_dataset):
    '''
    Cleans, imputes, scales, and encodes a dataset of NBA player stats.

    This function drops irrelevant columns, imputes missing numeric values using the median,
    applies standard scaling to numeric columns, and one-hot encodes the 'Position' column.

    Parameters
    ----------
    player_dataset : DataFrame
        Raw dataset of player stats, salary, and categorical columns.

    Returns
    -------
    DataFrame
        Cleaned and transformed dataset ready for modeling.
    '''

    
    # Convert to dataset df
    player_data_df = pd.DataFrame(player_dataset) 

    # sometimes first column is Unnamed, if it is rename to index to drop
    if 'Unnamed: 0' in player_data_df.columns:
        player_data_df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)   
    # Drop Index, Player Name, and Team since they don't contrubute to salary
    player_data_cleaned = player_data_df.drop(['Index', 'Player Name', 'Team'], axis='columns')

    
    # split data by data types
    numeric_cols = player_data_cleaned.select_dtypes(include='number').columns
    categorical_cols = ['Position']  # only non-numeric column we're keeping
    
    # numeric pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    #  categorical pipeline
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder())
    ])
    
    # Combine with ColumnTransformer
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    # process the new dataset
    processed_data = full_pipeline.fit_transform(player_data_cleaned)

    # Construct DataFrame with proper column names
    encoded_feature_names = full_pipeline.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    all_feature_names = list(numeric_cols) + list(encoded_feature_names)
        
    return pd.DataFrame(processed_data, columns=all_feature_names)