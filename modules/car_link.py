import pandas as pd
import sys
sys.path.append('..')
from modules.car_scraping import cars_links_generator, scrape_used_cars_data
from modules.car_cleanser import clean_my_car, brand_my_car, paint_my_car, car_dtype
from modules.car_merger import group_cars, order_att

def used_car_input(url):
    '''
    Obtains the data for the used car to predict the final price.
    :url: link to the used car to be analysed
    '''
    url_list = []
    url_list.append(url)
    car_data = scrape_used_cars_data([url])
    
    cols = ['title', 'price', 'year', 'kms', 'city', 'gearbox', 'doors', 'seats', 'power',
            'color', 'co2_emiss', 'fuel_type', 'warranty', 'dealer', 'chassis', 'height',
            'length', 'width', 'trunk_vol', 'max_speed', 'urban_cons', 'xtrurban_cons',
            'mixed_cons', 'weight', 'tank_vol', 'acceleration']
    
    df_scrap = pd.DataFrame(car_data, columns=cols).drop(0)
    
    df_scrap = clean_my_car(df_scrap)
    df_scrap = brand_my_car(df_scrap)
    df_scrap = paint_my_car(df_scrap)
    df_scrap = order_att(df_scrap)
    df_scrap = car_dtype(df_scrap)
    df_scrap['type'] = 'other'
    
    df = df_scrap.drop(['title', 'urban_cons', 'xtrurban_cons', 'price'], axis=1)
    y = df_scrap['price']
    
    return (df, y)
