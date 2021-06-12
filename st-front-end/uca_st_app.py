import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import sys
sys.path.append('/home/dsc/Dropbox/UsedCarsAppraiser')
from modules.fe_cars import frontend_preproc
from modules.pickle_jar import decompress_pickle
import shap
import matplotlib.pyplot as plt
import seaborn

st.header("Used Cars **App**raisser")
st.write("""
Thinking about buying a second-hand car in Spain? This application is based on a ML CatBoost algorithm. The model was trained using a dataset of 55,326 real second-hand cars from [coches.com](https://www.coches.com/). It can predict prices of used cars up to 100,000€ in the Spanish market.
""")
st.write('---')

# Load Cars dataset
cars = pd.read_csv('data/cleaned_cars.csv')
X = cars.drop(['title', 'price'], axis=1)
y = cars['price']

# Sidebar
st.sidebar.header('Specify Input Features')

def user_input_features():
    """
    Generates a DataFrame with all the inputs that the user did to make a prediction
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
    YEAR = st.sidebar.slider('Year', X.year.min(), X.year.max(), int(round(X.year.mean(),0)))
    KMS = st.sidebar.number_input('Kms', 0, 500000, int(round(X.kms.mean(),0)), help='Select a value between 0 and 500,000')
    DOORS = st.sidebar.slider('Doors', X.doors.min(), X.doors.max(), int(round(X.doors.mean(),0)))
    SEATS = st.sidebar.slider('Seats', X.seats.min(), X.seats.max(), int(round(X.seats.mean(),0)))
    POWER = st.sidebar.slider('Power', X.power.min(), X.power.max(), int(round(X.power.mean(),0)))
    CO2 = st.sidebar.slider(u'CO\u2082 emissions', X.co2_emiss.min(), X.co2_emiss.max(), int(round(X.co2_emiss.mean(),0)))
    HEIGHT = st.sidebar.slider('Height', X.height.min(), X.height.max(), int(round(X.height.mean(),0)))
    LENGTH = st.sidebar.slider('Length', X.length.min(), X.length.max(), int(round(X.length.mean(),0)))
    WIDTH = st.sidebar.slider('Width', X.width.min(), X.width.max(), int(round(X.width.mean(),0)))
    TRUNK = st.sidebar.slider('Trunk volume', X.trunk_vol.min(), X.trunk_vol.max(), int(round(X.trunk_vol.mean(),0)))
    SPEED = st.sidebar.slider('Max speed', X.max_speed.min(), X.max_speed.max(), int(round(X.max_speed.mean(),0)))
    CONS = st.sidebar.slider('Mixed consumption', X.mixed_cons.min(), X.mixed_cons.max(), X.mixed_cons.mean())
    WEIGHT = st.sidebar.slider('Weight', X.weight.min(), X.weight.max(), int(round(X.weight.mean(),0)))
    TANK = st.sidebar.slider('Tank volume', X.tank_vol.min(), X.tank_vol.max(), int(round(X.tank_vol.mean(),0)))
    ACC = st.sidebar.slider('Acceleration', X.acc.min(), X.acc.max(), X.acc.mean())
    
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

# Write input features set
df_input = user_input_features()
df = pd.concat([df_input, X], axis=0).reset_index().drop('index', axis=1)
st.subheader('User Inputs: Technical specs')
st.write(df_input)

# Applying feature engineering to the DataFrame before applying the model
df = frontend_preproc(df, y)

# Taking only first row after Feature Engineering to predict user's input
df_pred = df[:1]

# Load in model
cb_model = decompress_pickle("notebooks/models/cb_model.pbz2")

# Apply model to predict price
prediction = cb_model.predict(df_pred)
prediction = pd.DataFrame(prediction, columns=["Price prediction"])\
                .style.format('{:20,.2f}€')

st.write("The reasonable price for this second-hand car:")
st.write(prediction)
st.write('---')

# Explaining model's ouput predictions using SHAP API
df_shap = df[:500]
explainer = shap.Explainer(cb_model)
shap_values = explainer(df_shap)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.header('Feature Importance to the prediction')

plt.title('Feature importance to the model')
shap.plots.beeswarm(shap_values)
st.pyplot(bbox_inches='tight')

plt.title('Mean absolute value based on SHAP values for each feature')
shap.plots.bar(shap_values)
st.pyplot(bbox_inches='tight')
st.write('---')

# Final reference to the project
st.subheader('References')
st.write("""
For further details regarding this project, please refer to its [repo on GitHub](https://github.com/caresppen/UsedCarsAppraiser).
Here, you will be able to find all the scripts and notebooks used in dataset creation, analysis, visualizations and modeling. You can also download the models used in this app and use them for any other aims.

Created by Carlos Espejo Peña.
""")