import pandas as pd
import numpy as np
# from category_encoders import SamplingBayesianEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer

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


def frontend_preproc(df, y):
    '''
    Function that produces the preprocessing of the DataFrame before applying the model on the front-end.
    :df: concat of df_input by the user and X features of the model
    :y: target
    '''
    ### Feature Engineering
    ohe_cols = ['gearbox', 'fuel_type', 'warranty', 'dealer', 'doors']

    # OHE
    ohe = OneHotEncoder(categories='auto')
    feature_arr = ohe.fit_transform(df[ohe_cols]).toarray()
    feature_labels = ohe.categories_

    # Using a dictionary to produce all the new OHE columns
    feature_cols = []
    for k, v in dict(zip(ohe_cols, feature_labels)).items():
        for i in v:
            el = k + '_' + str(i)
            feature_cols.append(el)

    ohe_features = pd.DataFrame(feature_arr, columns=feature_cols)
    df = pd.concat([df, ohe_features], axis=1)
    df = df.drop(ohe_cols, axis=1)

    # Target Encoding
    cat_cols = df.select_dtypes(exclude=["number"]).columns
    cols_encoded = list(map(lambda c: c + '_encoded', cat_cols))

    t_encoder = TargetEncoder()
    t_encoder.fit(df[1:][cat_cols], y)
    df[cols_encoded] = t_encoder.transform(df[cat_cols])
    df = df.drop(cat_cols, axis=1)

    # Column Transformation: QuantileTransformer
    qt = QuantileTransformer(n_quantiles=500,
                             output_distribution='normal',
                             random_state=33)

    data = qt.fit_transform(df)
    df = pd.DataFrame(data, columns=df.columns)
    
    return df
