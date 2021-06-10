import pandas as pd
import numpy as np    

def clean_my_car(df):
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
    df['price'] = df.price.str.replace('.', '', regex=False).str.replace('€', '')
    df.loc[df.year.str.len() > 4, 'year'] = df.year.str[-4:]
    df['kms'] = df.kms.str.replace('.', '', regex=False).str.replace('km', '').str.replace('NUEVO', '0')
    df['power'] = df.power.str.replace(' cv', '')
    df['co2_emiss'] = df.co2_emiss.str.replace(' gr/m', '')
    df['height'] = df.height.str.replace(' cm', '')
    df['length'] = df.length.str.replace(' cm', '')
    df['width'] = df.width.str.replace(' cm', '')
    df[['trunk_vol']] = df.trunk_vol.str.replace(' l', '').str.replace('.', '', regex=False)
    df['max_speed'] = df.max_speed.str.replace(' km/h', '')
    df['urban_cons'] = df.urban_cons.str.replace(' l', '').str.replace(',', '.').astype(float)
    df['xtrurban_cons'] = df.xtrurban_cons.str.replace(' l', '').str.replace(',', '.').astype(float)
    df['mixed_cons'] = df.mixed_cons.str.replace(' l', '').str.replace(',', '.').astype(float)
    df['weight'] = df.weight.str.replace(' kg', '').str.replace('.', '', regex=False)
    df['tank_vol'] = df.tank_vol.str.replace(' l', '')
    df['acceleration'] = df.acceleration.str.replace(' s', '').str.replace(',', '.').astype(float)
    
    # Cars cannot have None doors.
    # In the EDA, Renault Twizy was detected to have 0 doors, when actually has 2.
    df['doors'] = df.doors.str[0].str.replace('0', '2')
    
    ### Cleaning cities column ###
    # Extract last two elements from the list. Reverse the order
    # Concat list in a string. Deleting unnecessary strings
    # Empty cities replaced by Unknown
    # Dealing with exceptions
    df['city'] = df.city.str.split(' ').str[-2:].str[::-1]\
                    .map(' '.join).str.replace(' en', '').str.replace(',', '', regex=False)\
                    .str.replace(r'(cv.*)', 'Unknown', regex=True)\
                    .str.replace('Real Ciudad', 'Ciudad Real').str.replace('Balears Illes', 'Baleares')
    
    # Translating Warranty, Dealer, Fuel_type columns
    df['warranty'] = df.warranty.str.replace('SÍ', 'YES').str.replace('No', 'NO')
    df['dealer'] = df.dealer.str.replace('Profesional', 'Professional').str.replace('Particular', 'Individual')
    df['fuel_type'] = df.fuel_type.str.replace('Gasolina', 'Gasoline')\
                                    .str.replace('Eléctrico', 'Electric')\
                                    .str.replace('Híbrido', 'Hybrid')
    
    # Translating gearbox column
    df = df.rename({'gear': 'gearbox'}, axis=1)
    df['gearbox'] = df.gearbox.replace('Manual automatizada', 'Manual')\
                            .replace('Automática continua, secuencial','Automatic')\
                            .replace('Directo, sin caja de cambios', 'Direct')\
                            .replace('Automática secuencial', 'Automatic')\
                            .replace('Automática continua', 'Automatic')\
                            .replace('Automática', 'Automatic')
    
    # Creating a dictionary to translate the chassis
    es_chassis = list(df.chassis.unique())

    try:
        es_chassis.remove('Chasis')
        es_chassis.append('Chasis') # Adding 'Chasis' by the end of the list
    except:
        print('No Chasis in the DataFrame')

    en_chassis = ['Convertible', 'Coupe', 'Roadster', 'Targa', 'Sedan', 'Sedan', 'Minivan', 'Combi', 'Van', 'Van', 'Bus', 'Offroad', 'Pickup', 'Pickup', 'Stationwagon', 'Van', 'Chassis']
    chassis_dict = dict(zip(es_chassis, en_chassis))

    for es, en in chassis_dict.items():
        df['chassis'] = df.chassis.str.replace(es, en)
    
    # Renaming columns
    df = df.rename({'acceleration': 'acc'}, axis=1)
    
    return df


def brand_my_car(df):
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
    
    # Conditional lists
    long_brands = ['Land Rover', 'Aston Martin', 'Alfa Romeo', 'Mercedes Amg']
    long_models = ['Coupé', 'Serie', 'Clase', 'Grand', 'Grande', 'Santa', 'Glory', 'Model', 'Xsara', 'Pt',
                   'Is', 'Es', 'Ct', 'Rx', 'Nx', 'Ux', 'Rc', 'Gs', 'Ls'] # Lexus

    df['title'] = df['title'].str.replace(r'\b(\w+)(\s+\1)+\b', r'\1', regex=True)

    # Excep: Volkswagen Up contains '!'
    df['title'] = df.title.str.replace('!', '')

    # Excep: E Advance
    df['title'] = df.title.str.replace('E Honda', 'E-Advance')

    # Splitting columns to conform both brands & models
    df[['brand', 'aux_1', 'aux_2', 'aux_3']] = df.title.str.split(' ', 3, expand=True)

    # Excep: Ds
    df['brand'] = df.brand.str.replace('Ds', 'Citroen')

    # Aux columns
    df['brand_long'] = df.brand + ' ' + df.aux_1
    df['model_long'] = df.aux_1 + ' ' + df.aux_2
    df['model'] = df['aux_1']

    df.loc[df.brand_long.isin(long_brands), 'brand'] = df['brand_long']
    df.loc[df.brand_long.isin(long_brands), 'model'] = df['aux_2']

    df.loc[df.aux_1.isin(long_models), 'model'] = df['model_long']

    # Deleting columns
    df.drop(['aux_1', 'aux_2', 'aux_3', 'brand_long', 'model_long'], axis=1, inplace=True)

    # Dealing with Exceptions
    df['model'] = df.model.str.replace('Grande ', '') # FIAT: 'Grande Punto' evolved to 'Punto'
    df['model'] = df.model.str.replace('1.6i', 'Coupé', regex=False) # Citroen: '1.6i' is aka 'Coupé'
    
    # Changing model/brand to uppercase
    df['model'] = df.model.str.upper()
    df['brand'] = df.brand.str.upper()

    return df


def paint_my_car(df):
    '''
    Function:
    - Cleans the dataset to obtain only general/common colors.
    - Translate colors from ES to EN.
    
    Parameters:
    * df = DataFrame to be cleaned
    '''
    
    list_numbers = [str(x) for x in list(range(0, 10))]
    dict_colors = {'BLANCO': 'WHITE', 'BLANC': 'WHITE', 'BIANCO': 'BLANCO', 'ALPINWEISS': 'WHITE',
                   'GRIS': 'GREY', 'GRAY': 'GREY',
                   'NEGRO': 'BLACK',
                   'AZUL': 'BLUE',
                   'ROJO': 'RED', 'GRANATE': 'RED', 'BURDEOS': 'RED',
                   'PLATA': 'SILVER', 'PLATEADO': 'SILVER',
                   'MARRÓN': 'BROWN', 'MARRON': 'BROWN',
                   'VERDE': 'GREEN',
                   'BEIGE': 'BEIGE',
                   'AZUL MARINO': 'NAVY BLUE',
                   'NARANJA': 'ORANGE',
                   'AMARILLO': 'YELLOW',
                   'BRONCE': 'BRONZE',
                   'VIOLETA': 'PURPLE', 'MORADO': 'PURPLE',
                   'ROSA': 'PINK',
                   'OTRO': 'OTHER'
                  }
    
    colors = sorted(df.color.dropna().unique())
    colors_up = [c.upper() for c in colors]
    
    # Remove numbers from colors
    df.loc[df.color.isin(list_numbers), 'color'] = 'OTHER'

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

    # Applying 'OTHER' to complex colors
    en_colors = list(dict_colors.values())
    df.loc[~df.color.isin(en_colors), 'color'] = 'OTHER'

    # Step 2: Translating colors into English & simplifying colors
    for es_c, en_c in dict_colors.items():
        df.loc[df.color.str.contains(es_c), 'color'] = en_c
        df.loc[df.color.str.contains(en_c), 'color'] = en_c

    return df


def car_dtype(df):
    '''
    Function:
    Changes the data types of the attributes of the input DataFrame.
    
    Parameters:
    * df = DataFrame to be changed
    '''
    df['year'] = df.year.astype(int)
    df['kms'] = df.kms.astype(int)
    df['doors'] = df.doors.astype(int)
    df['seats'] = df.seats.astype(int)
    df['power'] = df.power.astype(int)
    df['co2_emiss'] = df.co2_emiss.astype(int)
    df['height'] = df.height.astype(int)
    df['length'] = df.length.astype(int)
    df['width'] = df.width.astype(int)
    df['trunk_vol'] = df.trunk_vol.astype(int)
    df['max_speed'] = df.max_speed.astype(int)
    df['weight'] = df.weight.astype(int)
    df['tank_vol'] = df.tank_vol.astype(int)
    df['price'] = df.price.astype(int)
    
    return df


def rm_outliers(df):
    '''
    Function:
    Detect the outliers on the DataFrame and removes them.
    Consider only those cars with a price lower than 100,000 to avoid that outliers distort the prediction. 
    
    Parameters:
    * df = DataFrame to be cleaned
    '''
    df = df[df.price <= 100000]
    df = df[df.kms <= 500000]
    
    return df
    