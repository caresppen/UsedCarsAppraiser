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
    
    # Filling NaN values with the mean per category
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
    
    # Changing data formats after applying means
    cols_dec = ['urban_cons', 'xtrurban_cons', 'mixed_cons', 'acc']
    cols_format = ['kms', 'co2_emiss', 'height', 'length', 'width',
               'trunk_vol', 'max_speed', 'weight', 'tank_vol']
    
    for col in cols_dec:
        df[col] = round(df[col], 1)
    
    for col in cols_format:
        df[col] = df[col].astype(int)
    
    return df


def renting_na(df):
    '''
    Function:
    - Set the mean for the rest of cols based on 'brand' & 'model'.
    - Drop renting cars that could introduce noise in the analysis.
    
    Parameters:
    df: DataFrame to be cleaned in NaN values
    '''
    # Defining all 'zeros' as NaN
    df = df.mask(df==0)
    
    # Setting columns to apply the mean based on cars' models & brands
    cols_fill_zero = ['co2_emiss', 'trunk_vol', 'max_speed', 'urban_cons',
                      'xtrurban_cons', 'mixed_cons', 'tank_vol', 'acc']
    
    # Filling NaN values with the mean per category
    for col in cols_fill_zero:
        df[col] = df[col].fillna(df.groupby(['brand', 'model'])[col].transform('mean'))
        df[col] = df[col].fillna(df.groupby('brand')[col].transform('mean'))
    
    # Filling NaN values
    df = df.fillna(0)
    
    return df


def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)



