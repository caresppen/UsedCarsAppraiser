import pandas as pd
import numpy as np

def cars_na(df):
    '''
    Function:
    - Drop NaN values based on columns distinct from 'kms' & 'co2_emiss'.
    - Set the mean for the rest of cols based on 'brand' & 'model'.
    - Drop cars that could introduce noise in the analysis.

    Parameters:
    df: DataFrame to be cleaned in NaN values
    '''
    # Defining all 'zeros' as NaN
    df = df.mask(df==0)
    
    # Setting columns to apply the mean based on cars' models & brands
    cols_fill_zero = ['max_speed', 'height', 'length', 'width', 'trunk_vol', 'urban_cons',
                  'xtrurban_cons', 'mixed_cons', 'weight', 'tank_vol', 'acc']
    
    # Applying changes to the columns
    for col in cols_fill_zero:
        df[col] = df[col].fillna(df.groupby(['brand', 'model'])[col].transform('mean'))
        df[col] = df[col].fillna(df.groupby('brand')[col].transform('mean'))
    
    # Dealing with electric cars
    # Condition: year > 2018
    df.loc[df.year < 2018, 'co2_emiss'] = df['co2_emiss'].fillna(df.groupby(['brand', 'model'])['co2_emiss'].transform('mean'))
    
    # Defining columns with NaN value
    dropna_subset = df.columns[df.isna().any()].drop(['kms', 'co2_emiss'])
    
    # Dropping columns
    df = df.dropna(subset=dropna_subset)
    
    # Filling NaN values in 'kms' = km0 cars
    # Filling NaN values in 'co2_emiss' = electric cars
    df = df.fillna(0)
    
    return df
