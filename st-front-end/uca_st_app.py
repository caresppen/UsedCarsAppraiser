import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import sys
sys.path.append('/home/dsc/Dropbox/UsedCarsAppraiser')
from modules.fe_cars import frontend_preproc
from modules.pickle_jar import decompress_pickle

def user_input_features(X):
    """
    Generates a DataFrame with all the inputs that the user did to make a prediction.
    New add-in: Read a coches.com input URL from the user
    """
    BRAND = st.sidebar.selectbox('Brand', np.sort(X.brand.unique()), index=int(np.where(np.sort(X.brand.unique())=='VOLVO')[0][0]), help='Choose car brand')
    MODEL = st.sidebar.selectbox('Model', np.sort(X[X.brand == BRAND].model.unique()), index=int(len(X[X.brand == BRAND].model.unique())/2), help='Models available for the selected brand')
    TYPE = st.sidebar.selectbox('Type', X.type.unique(), index=int(np.where(X.type.unique()=='medium')[0][0]))
    CITY = st.sidebar.selectbox('City', X.city.unique(), index=int(np.where(X.city.unique()=='Sevilla')[0][0]))
    GEARBOX = st.sidebar.selectbox('Gearbox', X.gearbox.unique(), index=int(np.where(X.gearbox.unique()=='Manual')[0][0]))
    COLOR = st.sidebar.selectbox('Color', X.color.unique(), index=int(np.where(X.color.unique()=='WHITE')[0][0]))
    FUEL = st.sidebar.selectbox('Fuel', X.fuel_type.unique(), index=int(np.where(X.fuel_type.unique()=='Gasoline')[0][0]))
    WARRANTY = st.sidebar.selectbox('Warranty', X.warranty.unique())
    DEALER = st.sidebar.selectbox('Dealer', X.dealer.unique())
    CHASSIS = st.sidebar.selectbox('Chassis', X.chassis.unique(), index=int(np.where(X.chassis.unique()=='Sedan')[0][0]))
    YEAR = st.sidebar.slider('Year', int(X.year.min()), int(X.year.max()), int(round(X.year.mean(),0)))
    KMS = st.sidebar.number_input('Kms', 0, 500000, int(round(X.kms.mean(),0)), help='Select a value between 0 and 500,000')
    DOORS = st.sidebar.slider('Doors', int(X.doors.min()), int(X.doors.max()), int(round(X.doors.mean(),0)))
    SEATS = st.sidebar.slider('Seats', int(X.seats.min()), int(X.seats.max()), int(round(X.seats.mean(),0)))
    POWER = st.sidebar.slider('Power', int(X.power.min()), int(X.power.max()), int(round(X.power.mean(),0)))
    CO2 = st.sidebar.slider(u'CO\u2082 emissions', int(X.co2_emiss.min()), int(X.co2_emiss.max()), int(round(X.co2_emiss.mean(),0)))
    HEIGHT = st.sidebar.slider('Height', int(X.height.min()), int(X.height.max()), int(round(X.height.mean(),0)))
    LENGTH = st.sidebar.slider('Length', int(X.length.min()), int(X.length.max()), int(round(X.length.mean(),0)))
    WIDTH = st.sidebar.slider('Width', int(X.width.min()), int(X.width.max()), int(round(X.width.mean(),0)))
    TRUNK = st.sidebar.slider('Trunk volume', int(X.trunk_vol.min()), int(X.trunk_vol.max()), int(round(X.trunk_vol.mean(),0)))
    SPEED = st.sidebar.slider('Max speed', int(X.max_speed.min()), int(X.max_speed.max()), int(round(X.max_speed.mean(),0)))
    CONS = st.sidebar.slider('Mixed consumption', float(X.mixed_cons.min()), float(X.mixed_cons.max()), float(X.mixed_cons.mean()))
    WEIGHT = st.sidebar.slider('Weight', int(X.weight.min()), int(X.weight.max()), int(round(X.weight.mean(),0)))
    TANK = st.sidebar.slider('Tank volume', int(X.tank_vol.min()), int(X.tank_vol.max()), int(round(X.tank_vol.mean(),0)))
    ACC = st.sidebar.slider('Acceleration', float(X.acc.min()), float(X.acc.max()), float(X.acc.mean()))
    
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
    
    PAGE_CONFIG = {"page_title": "UCApp",
                   "page_icon": "https://www.pngkey.com/png/full/366-3662307_to-get-started-please-fill-out-your-information.png",
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
    
    # White: https://biddown.com/wp-content/uploads/2020/03/4-41860_white-wallpaper-background-full-hd-background-white.jpg
    # Google Drive: https://drive.google.com/uc?export=view&id=1ZlLnr5nLLTGBYA7xWpE3picPEIsMTWCE
    
    # Icon: https://www.pngkey.com/png/full/366-3662307_to-get-started-please-fill-out-your-information.png
    # Bugatti: https://pngimg.com/uploads/bugatti/bugatti_PNG31.png
    # AUDI: https://image.flaticon.com/icons/png/512/741/741460.png
    # BMW: https://img-premium.flaticon.com/png/512/741/741403.png?token=exp=1623588793~hmac=189aba02502ff104d954d153cb7e675a
    # Normal Red: https://img-premium.flaticon.com/png/512/3085/3085330.png?token=exp=1623588897~hmac=4c4646b5fc8ae43fbfdb648781b25741
    # BG Car: https://image.flaticon.com/icons/png/512/1040/1040634.png
    # Normal Blue: https://img-premium.flaticon.com/png/512/1048/1048314.png?token=exp=1623588933~hmac=974cdbfb2e58bb3c095e7ffbcb62fded
    # Appraisal: https://img-premium.flaticon.com/png/512/4856/4856356.png?token=exp=1623589774~hmac=36396ae2b63ac6de8778a47610c89741
    # Go: https://img-premium.flaticon.com/png/512/4856/4856364.png?token=exp=1623589856~hmac=f59c012b85676d66f381344906f0421c
    # Wheel: https://image.flaticon.com/icons/png/512/3003/3003735.png
    # Mario: https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/i/c8a2987f-3e2e-4e4d-9aac-27d02b24bdd3/d6th2q4-8588e7b0-13bb-45e2-b881-51115b5d03a3.png


def html_header(url):
     st.markdown(f'<b style="color:#439AD6;font-size:32px;">{url}</b>', unsafe_allow_html=True)


def html_text(url):
     st.markdown(f'<p style="background-color:#FFFFFF;color:#384B8F;">{url}</p>', unsafe_allow_html=True)

    
def main():
    
    # Set page parameters
    page_params()
    
    html_header("Used Cars APPraisser")
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

    # Write input features set
    df_input = user_input_features(X)
    df = pd.concat([df_input, X], axis=0).reset_index().drop('index', axis=1)
    st.subheader(':computer: User Inputs: Technical specs')
    st.dataframe(df_input)
    
    # Applying feature engineering to the DataFrame before applying the model
    df = frontend_preproc(df, y)

    # Taking only first row after Feature Engineering to predict user's input
    df_pred = df[:1]

    # Load in model
    cb_model = decompress_pickle("notebooks/models/cb_model.pbz2")

    # Apply model to predict price
    st.subheader(":crystal_ball: Prediction")
    prediction = cb_model.predict(df_pred)
    prediction = pd.DataFrame(prediction, columns=["Price prediction"])\
                    .style.format('{:20,.2f}€')

    st.write("The reasonable price for this second-hand car is:")
    st.dataframe(prediction)
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
    
    Having a deeper look into the values, we can conclude that the higher the price of the car, the more dispersed is the model. The number of second-hand cars with a value higher than **50,000 €** are scarce compared to more accesible cars. Therefore, we could expect a better performance of the model when predicting prices of non-luxury cars.
    ''')
    st.image(Image.open('notebooks/fig/12_model_pred_cb.png'))
    
    st.write('---')

    # Final reference to the project
    st.subheader(':link: References')
    st.write("""
    For further details regarding this project, please refer to its [repo on GitHub](https://github.com/caresppen/UsedCarsAppraiser).
    Here, you will be able to find all the scripts and notebooks used in dataset creation, analysis, visualizations and modeling. You can also download the models used in this app and use them for any other aims.

    Created by Carlos Espejo Peña
    
    Contact: [![LinkedIn](https://drive.google.com/uc?export=view&id=1nx0u9GeUyYttqyju6Z1824UCqto6hXZv)](https://www.linkedin.com/in/carlosespejopena/) [![GitHub](https://drive.google.com/uc?export=view&id=17_77FAziJKdyZaRkjzlGFTKaPAKGdszl)](https://github.com/caresppen)
    """)

if __name__ == '__main__':
    main()