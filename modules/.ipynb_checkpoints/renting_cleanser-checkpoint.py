import pandas as pd
import numpy as np

def brand_renting(df):
    '''
    Function:
    - Splitting columns: i.e. titles into car brands/models.
    - Dealing with exceptions to produce brands/models.
    
    # regex expression needed to remove duplicated words in titles:
    \b        # word boundary
    (\w+)     # 1st capture group of a single word
    ( 
    \s+       # 1 or more spaces
    \1        # reference to first group 
    )+        # one or more repeats
    \b
    
    Parameters:
    * df = DataFrame to obtain the car's brand/model
    '''
    
    # regex applied to title
    df['title'] = df['title'].str.replace(r'\b(\w+)(\s+\1)+\b', r'\1', regex=True)

    # Splitting columns to conform both brands & models
    df[['brand', 'model']] = df.title.str.split(' ', 1, expand=True)

    # Excep: DS
    df['brand'] = df.brand.str.replace('DS', 'CITROEN')
    
    # Applying uppercases
    df['model'] = df.model.str.upper()
    
    return df



def clean_renting(df):
    '''
    Function:
    Cleans the entire cars dataset. It executes the following tasks:
    - Numerical columns: Selecting only the int/float part of each of the columns that contains numbers.
    - ES>EN Translations: The source datasets are in Spanish. Translates every value to English.
    - Columns standardization: Setting buckets to allocate all the components into a major group.
    
    Parameters:
    * df = DataFrame to be cleaned
    '''
    
    # Extracting numbers from all the Qunatitative columns
    df['km_year'] = df.km_year.astype(int) * 1000
    df['power'] = df.power.str.split('cv').str[0]
    df['co2_emiss'] = df.co2_emiss.str.replace(' Co gr/km', '')
    df['height'] = df.height.str.replace(' cm', '').str.replace(',', '.').astype(float)
    df['length'] = df.length.str.replace(' cm', '').str.replace(',', '.').astype(float)
    df['width'] = df.width.str.replace(' cm', '').str.replace(',', '.').astype(float)
    df[['trunk_vol']] = df.trunk_vol.str.replace(' l', '')
    df['max_speed'] = df.max_speed.str.replace(' km/h', '')
    
    df['urban_cons'] = df.urban_cons.str.replace(' l', '').str.replace(',', '.').astype(float)
    df['xtrurban_cons'] = df.xtrurban_cons.str.replace(' l', '').str.replace(',', '.').astype(float)
    df['mixed_cons'] = df.mixed_cons.str.replace(' l', '').str.replace(',', '.').astype(float)
    df['weight'] = df.weight.str.replace(' kg', '').str.replace('.', '', regex=False)
    df['tank_vol'] = df.tank_vol.str.replace(' l', '')
    df['acceleration'] = df.acceleration.str.replace(' s', '').str.replace(',', '.').astype(float)
    
    # Translating Warranty, Maintenance, Status, Fuel_type, Tires columns
    df['warranty'] = df.warranty.str.replace('Incluido', 'Included')
    df['tires'] = df.tires.str.replace('Incluido', 'Included')\
                                .replace('No incluido', 'Not Included')
    df['maintenance'] = df.maintenance.str.replace('Incluido', 'Included')\
                                    .replace('No incluido', 'Not Included')
    df['status'] = df.status.str.replace('Nuevo', 'New')\
                                    .replace('Seminuevo', 'Preowned')
    df['fuel_type'] = df.fuel_type.str.replace('Gasolina', 'Gasoline')\
                                    .replace('Eléctrico', 'Electric')\
                                    .replace('Híbrido', 'Hybrid')
    
    # Translating gearbox column
    df = df.rename({'gear': 'gearbox'}, axis=1)
    df['gearbox'] = df.gearbox.str.replace('Manual automatizada', 'Manual')\
                            .str.replace('Automática continua, secuencial','Automatic')\
                            .str.replace('Directo, sin caja de cambios', 'Direct')\
                            .str.replace('Automática secuencial', 'Automatic')\
                            .str.replace('Automática continua', 'Automatic')\
                            .str.replace('Automática', 'Automatic')
    
    # Creating a dictionary to translate the chassis
    es_chassis = list(df.chassis.unique())
    en_chassis = ['Sedan', 'Offroad', 'Van', 'Coupe', 'Stationwagon', 'Minivan', 'Combi']
    chassis_dict = dict(zip(es_chassis, en_chassis))

    for es, en in chassis_dict.items():
        df['chassis'] = df.chassis.str.replace(es, en)
    
    # change long col name: contract, acceleration
    df = df.rename({'contract_months': 'c_months',
                    'acceleration': 'acc'},
                   axis=1)
    
    return df



def paint_renting(df):
    '''
    Function:
    - Cleans the dataset to obtain only general/common colors.
    - Translate colors from ES to EN.
    
    Parameters:
    * df = DataFrame to be cleaned
    '''
    
    dict_colors = {'BLANCO': 'WHITE',
                   'GRIS': 'GREY',
                   'NEGRO': 'BLACK',
                   'AZUL': 'BLUE',
                   'ROJO': 'RED', 'GRANATE': 'ROJO',
                   'PLATEADO': 'SILVER',
                   'NARANJA': 'ORANGE',
                   'OTRO': 'OTHER', 'CONSULTAR': 'OTHER' # To be consulted = Other/NotDefined
                  }

    # Assigning 'OTHER' to empty colors
    df.loc[df.color.isna(), 'color'] = 'OTHER'

    # Transforming color column to upper
    df['color'] = [c.upper() for c in df.color]

    # Stripping colors
    df['color'] = df.color.str.strip()

    # Step 1: Translating colors into English & simplifying colors
    for es_c, en_c in dict_colors.items():
        df.loc[df.color.str.contains(es_c), 'color'] = en_c
        df.loc[df.color.str.contains(en_c), 'color'] = en_c

    return df



def order_typify(df):
    '''
    Function:
    Order Attributes function to get the desired car format to be explored.
    
    Parameters:
    * df = DataFrame which columns need to be ordered.
    '''
    # Defining final column order
    col_order = ['title', 'brand', 'model', 'c_months', 'km_year', 'fuel_type',
                 'color', 'gearbox', 'doors', 'seats', 'warranty', 'maintenance',
                 'tires', 'status', 'chassis',  'power', 'co2_emiss', 'height',
                 'length', 'width', 'trunk_vol', 'max_speed', 'urban_cons',
                 'xtrurban_cons', 'mixed_cons', 'weight', 'tank_vol', 'acc',
                 'price']

    df = df.reindex(columns=col_order)
    
    # Selecting cols type
    df['power'] = df.power.astype('int')
    df['co2_emiss'] = df.co2_emiss.astype('int')
    df['trunk_vol'] = df.trunk_vol.astype('int')
    df['max_speed'] = df.max_speed.astype('int')
    df['weight'] = df.weight.astype('int')
    df['tank_vol'] = df.tank_vol.astype('int')
    df['acc'] = df.acc.astype('float')
    
    return df
