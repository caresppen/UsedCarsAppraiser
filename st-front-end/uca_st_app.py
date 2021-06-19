import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import sys
sys.path.append('/home/dsc/Dropbox/UsedCarsAppraiser')
from modules.fe_cars import frontend_preproc
from modules.pickle_jar import decompress_pickle
from modules.car_link import used_car_input
from modules.frontend_fmt import footer, html_header

def user_input_features(data, X):
    """
    Generates a DataFrame with all the inputs that the user did to make a prediction.
    New add-in: Read a coches.com input URL from the user
    :data: input parameters of the car from the URL. If no URL inserted, the default model will be displayed.
    :X: set of data with the model max & min to configure the car to be predicted.
    """
    if data.shape[0] > 1:
        BRAND = st.sidebar.selectbox('Brand', np.sort(X.brand.unique()), index=int(np.where(np.sort(X.brand.unique())=='VOLVO')[0][0]), help='Choose car brand')
        MODEL = st.sidebar.selectbox('Model', np.sort(X[X.brand == BRAND].model.unique()), index=int(len(X[X.brand == BRAND].model.unique())/2), help='Models available for the selected brand')
        TYPE = st.sidebar.selectbox('Type', X.type.unique(), index=int(np.where(X.type.unique()=='medium')[0][0]))
        CITY = st.sidebar.selectbox('City', X.city.unique(), index=int(np.where(X.city.unique()=='Sevilla')[0][0]))
        GEARBOX = st.sidebar.selectbox('Gearbox', X.gearbox.unique(), index=int(np.where(X.gearbox.unique()=='Manual')[0][0]))
        COLOR = st.sidebar.selectbox('Color', X.color.unique(), index=int(np.where(X.color.unique()=='WHITE')[0][0]))
        FUEL = st.sidebar.selectbox('Fuel', X.fuel_type.unique(), index=int(np.where(X.fuel_type.unique()=='Gasoline')[0][0]))
        CHASSIS = st.sidebar.selectbox('Chassis', X.chassis.unique(), index=int(np.where(X.chassis.unique()=='Sedan')[0][0]))
    else:
        BRAND = st.sidebar.selectbox('Brand', np.sort(data.brand.unique()), index=0, help='Choose car brand')
        MODEL = st.sidebar.selectbox('Model', np.sort(data[data.brand == BRAND].model.unique()), index=0, help='Models available for the selected brand')
        TYPE = st.sidebar.selectbox('Type', data.type.unique(), index=0)
        CITY = st.sidebar.selectbox('City', data.city.unique(), index=0)
        GEARBOX = st.sidebar.selectbox('Gearbox', data.gearbox.unique(), index=0)
        COLOR = st.sidebar.selectbox('Color', data.color.unique(), index=0)
        FUEL = st.sidebar.selectbox('Fuel', data.fuel_type.unique(), index=0)
        CHASSIS = st.sidebar.selectbox('Chassis', data.chassis.unique(), index=0)
    
    WARRANTY = st.sidebar.selectbox('Warranty', X.warranty.unique())
    DEALER = st.sidebar.selectbox('Dealer', X.dealer.unique())
    YEAR = st.sidebar.slider('Year', int(X.year.min()), int(X.year.max()), int(round(data.year.mean(),0)))
    KMS = st.sidebar.number_input('Kms', 0, 500000, int(round(data.kms.mean(),0)), help='Select a value between 0 and 500,000')
    DOORS = st.sidebar.slider('Doors', int(X.doors.min()), int(X.doors.max()), int(round(data.doors.mean(),0)))
    SEATS = st.sidebar.slider('Seats', int(X.seats.min()), int(X.seats.max()), int(round(data.seats.mean(),0)))
    POWER = st.sidebar.slider('Power', int(X.power.min()), int(X.power.max()), int(round(data.power.mean(),0)))
    CO2 = st.sidebar.slider(u'CO\u2082 emissions', int(X.co2_emiss.min()), int(X.co2_emiss.max()), int(round(data.co2_emiss.mean(),0)))
    HEIGHT = st.sidebar.slider('Height', int(X.height.min()), int(X.height.max()), int(round(data.height.mean(),0)))
    LENGTH = st.sidebar.slider('Length', int(X.length.min()), int(X.length.max()), int(round(data.length.mean(),0)))
    WIDTH = st.sidebar.slider('Width', int(X.width.min()), int(X.width.max()), int(round(data.width.mean(),0)))
    TRUNK = st.sidebar.slider('Trunk volume', int(X.trunk_vol.min()), int(X.trunk_vol.max()), int(round(data.trunk_vol.mean(),0)))
    SPEED = st.sidebar.slider('Max speed', int(X.max_speed.min()), int(X.max_speed.max()), int(round(data.max_speed.mean(),0)))
    CONS = st.sidebar.slider('Mixed consumption', float(X.mixed_cons.min()), float(X.mixed_cons.max()), float(data.mixed_cons.mean()))
    WEIGHT = st.sidebar.slider('Weight', int(X.weight.min()), int(X.weight.max()), int(round(data.weight.mean(),0)))
    TANK = st.sidebar.slider('Tank volume', int(X.tank_vol.min()), int(X.tank_vol.max()), int(round(data.tank_vol.mean(),0)))
    ACC = st.sidebar.slider('Acceleration', float(X.acc.min()), float(X.acc.max()), float(data.acc.mean()))
    
    data = {'brand': BRAND,
            'model': MODEL,
            'type': TYPE,
            'year': YEAR,
            'kms': KMS,
            'city': CITY,
            'gearbox': GEARBOX,
            'doors': DOORS,
            'seats': SEATS,
            'power': POWER,
            'color': COLOR,
            'co2_emiss': CO2,
            'fuel_type': FUEL,
            'warranty': WARRANTY,
            'dealer': DEALER,
            'chassis': CHASSIS,
            'height': HEIGHT,
            'length': LENGTH,
            'width': WIDTH,
            'trunk_vol': TRUNK,
            'max_speed': SPEED,
            'mixed_cons': CONS,
            'weight': WEIGHT,
            'tank_vol': TANK,
            'acc': ACC,
           }
    
    features = pd.DataFrame(data, index=[0])
    
    return features


def page_params():
    
    PAGE_CONFIG = {"page_title": "Carlyst",
                   "page_icon": "https://drive.google.com/uc?export=view&id=1PkZx6L1SkT-DGRjsv-SOHHiF2vEL8yOW",
                   "layout": "centered",
                   "initial_sidebar_state": "auto"}
    
    st.set_page_config(**PAGE_CONFIG)
    
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://drive.google.com/uc?export=view&id=1ZlLnr5nLLTGBYA7xWpE3picPEIsMTWCE");
        background-size: cover;
    }
    </style>
    '''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Google Drive: https://drive.google.com/uc?export=view&id=1R4eNDfMno8ToYNYRTYW6JppCjEP-wpf9
    # BGWhite: https://biddown.com/wp-content/uploads/2020/03/4-41860_white-wallpaper-background-full-hd-background-white.jpg
    # Icon: https://www.pngkey.com/png/full/366-3662307_to-get-started-please-fill-out-your-information.png
    
def main():
    
    # Set page parameters
    page_params()
    
    # Title logo
    st.image("https://drive.google.com/uc?export=view&id=1cf6kuyI7QMg5VDDLlsbfL-Yrpi-OF_ap", width=None)  #685
    st.write("""
    Thinking about buying a second-hand car in Spain? This application will help you find a fair price for your desired car. The model, which is based on a ML CatBoost algorithm, was trained using a dataset of 55,326 real second-hand cars from [coches.com](https://www.coches.com/). It can predict prices of used cars up to 100,000€ in the Spanish market.
    """)
    st.write('---')

    # Load Cars dataset
    cars = pd.read_csv('data/cleaned_cars.csv')
    X = cars.drop(['title', 'price'], axis=1)
    y = cars['price']
    
    # Sidebar
    st.sidebar.header(':gear: Specify Input Parameters')
    
    # Input Sidebar link from coches.com
    st.sidebar.write('#### Look for your car at coches.com [![loupe](https://drive.google.com/uc?export=view&id=1vdDiJ8P5-rzTPmVnxyG1hdRWI5sr8Htk)](https://www.coches.com/coches-segunda-mano/coches-ocasion.htm)')
    url = st.sidebar.text_input('Insert a link to get a recommendation', help='Once you have a link of a second-hand car from coches.com, paste it in the input box. When no link is specified, a default car will appear.')
    st.sidebar.write('---')
    
    # Write input features set on Sidebar
    flag = 0    # flag to control the output dataframe for predicting the price
    try:
        car_link, y_link = used_car_input(url)
        df_input = user_input_features(car_link, X)
        flag = 1
    except:
        df_input = user_input_features(X, X)
        flag = 0
    
    df = pd.concat([df_input, X], axis=0).reset_index().drop('index', axis=1)
    st.subheader(':computer: User Inputs: Technical specs')
    st.dataframe(df_input.style.set_properties(**{'background-color': 'white'}))
    
    # Applying feature engineering to the DataFrame before applying the model
    df = frontend_preproc(df, y)

    # Taking only first row after Feature Engineering to predict user's input
    df_pred = df[:1]

    # Load in model
    cb_model = decompress_pickle("notebooks/models/cb_model.pbz2")

    # Apply model to predict price
    st.subheader(":crystal_ball: Prediction")
    st.write('When a link is input, savings and a recommendation about the purchase will appear.')
    prediction = cb_model.predict(df_pred)
    
    # Define final output table
    def red_green_cond_fmt(number):
        color = 'red' if number < 0 else 'green'
        return f'color: {color}; background-color: white'
    
    if flag == 1:
        df_pred = pd.DataFrame(prediction, columns=["Price prediction"])
        df_pred['Price on website'] = y_link.iloc[0]
        df_pred['Savings'] = df_pred['Price prediction'] - df_pred['Price on website']
        df_pred['Recommended purchase?'] = 'Yes' if df_pred['Savings'].iloc[0]>=0 else 'No'
        sv_color = '#C81C3C' if df_pred['Savings'].iloc[0]<=0 else 'green'
        
        for col in ['Price prediction', 'Price on website', 'Savings']:
            df_pred[col] = df_pred[col].apply(lambda x: '{:20,.0f}€'.format(x))

        df_pred = df_pred.T
        df_pred.columns = ['Result']

        df_pred = df_pred.style\
                         .set_properties(subset = pd.IndexSlice['Price prediction', :],
                                         **{'background-color': 'white', 'font-weight': 'bold'})\
                         .set_properties(subset = pd.IndexSlice['Savings', :],
                                         **{'color': sv_color})\
                         .set_properties(subset = pd.IndexSlice[['Price on website', 'Savings', 'Recommended purchase?'], :],
                                         **{'background-color': 'white'})\
                         .set_table_styles([{'selector': 'th',
                                             'props': [('column-width', '80px')]
                                            }
                                           ]
                                          )
        
    else:
        df_pred = pd.DataFrame(prediction, columns=["Price prediction"]).T
        df_pred.columns = ['Result']
        df_pred = df_pred.style.format('{:20,.0f}€').set_properties(**{'background-color': 'white', 'font-weight': 'bold'})
    
    # DataFrame prediction output
    st.dataframe(df_pred)
    st.write('---')
    
    # Explaining model's ouput predictions using SHAP plotted values
    st.write("## **Behind the scenes...**")
    st.write("This section is dedicated to every individual curious about the machine learning model that powers this tool.")
    st.subheader(':bar_chart: Parameters impact')
    st.write('''SHAP plots show the distribution of the impacts each parameter has on the final price prediction. The color represents the parameters values (red is high, blue is low).
    
This explains for example that a low manufactured year, lowers the final predicted car price. What is more, cars with a high horse power, will result on a higher prection. Finally, it is possible to conclude that the higher the total number of kilometers of the car, the lower the price.
    ''')
    st.image(Image.open('notebooks/fig/12_shap_model_distribution.png'))
    
    st.subheader(':chart_with_upwards_trend: How different are predictions from actual values?')
    st.write('''
    Below we can find a scatter plot of the actual vs predicted values of the model. The diagonal line shows the perfect regressor. Therefore, the closer all of the predictions are to this line, the better the model.
    
    Having a deeper look into the values, we can conclude that the higher the price of the car, the more dispersed is the model. The number of second-hand automobiles with a value higher than **50,000 €** are scarce compared to more accesible cars. Therefore, we could expect a better performance of the model when predicting prices of non-luxury brands.
    ''')
    st.image(Image.open('notebooks/fig/12_model_pred_cb.png'))
    
    st.write('---')
    
    # Final reference to the project
    st.subheader(':link: References')
    st.write("""
    For further details regarding this project, please refer to its [repo on GitHub](https://github.com/caresppen/UsedCarsAppraiser).
    Here, you will be able to find all the scripts and notebooks used in dataset creation, analysis, visualizations and modeling. You can also download the models used in this app and use them for any other aims.
    """)
    
    # Setting app footer
    footer()

if __name__ == '__main__':
    main()