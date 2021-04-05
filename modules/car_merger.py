import pandas as pd
import numpy as np
import os
import re

def group_cars(data_dir):
    '''
    Function:
    - Merging all datasets into one single dataframe.
    - Final dataframe contains the type of car: offroad, minivan, familiar, ...
    - Reorder columns to have an easier overview of the data.
    
    Parameters:
    * data_dir = directory wich contains all the csv's to be merged.
    '''
    
    # Setting a list with all the csv to be read and appended
    files = os.listdir(data_dir)
    
    # Filtering by the unnecessary files: using regex
    regex = re.compile(r'renting.*|.ipynb.*|merged_cars.csv|cars.csv')
    sel_files = [i for i in files if not regex.match(i)]
    
    # Moving km0 & used cars files by the end of the list 
    km0 = sel_files.pop(sel_files.index('km0_cars.csv'))
    used = sel_files.pop(sel_files.index('used_cars.csv'))
    sel_files = sel_files + [km0] + [used]
    
    # Compiling all the files inside the same df
    df = pd.DataFrame() # init df: to append the rest of the dataframes
    for file in sel_files:
        path = data_dir + file
        df_aux = pd.read_csv(path)
        df_aux['type'] = file.split('_')[0]
        df = df.append(df_aux)
        df = df.drop_duplicates(keep='first',
                                subset=df.columns.difference(['type']))
    
    # Replacing some values to std the df
    df['type'] = df.type.replace({'fam': 'familiar',
                                  'km0': 'other',
                                  'used': 'other'})
    
    # Resetting index after append
    df = df.reset_index().drop('index', axis=1)
    
    return df

    
def order_att(df):
    '''
    Function:
    Order Attributes function to get the desired car format to be explored.
    
    Parameters:
    * df = DataFrame which columns need to be ordered.
    '''
    # Defining final column order
    col_order = ['title', 'brand', 'model', 'type', 'year', 'kms', 'city', 'gearbox',
                 'doors', 'seats', 'power', 'color', 'co2_emiss', 'fuel_type',
                 'warranty', 'dealer', 'chassis', 'height', 'length', 'width',
                 'trunk_vol', 'max_speed', 'urban_cons', 'xtrurban_cons',
                 'mixed_cons', 'weight', 'tank_vol', 'acc', 'price']

    df = df.reindex(columns=col_order)
    
    return df