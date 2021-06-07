import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import sys
sys.path.append('/home/dsc/Dropbox/UsedCarsAppraiser')
from modules.pickle_jar import decompress_pickle
import shap
import matplotlib.pyplot as plt
import seaborn

st.header("Used Cars **App**raisser")
st.write("""
This application is based on a ML Random Forest algorithm. The model was trained using a dataset of 55,366 real second-hand cars from [coches.com](https://www.coches.com/). It can predict prices of used cars up to 100,000€
""")
st.write('---')

# Load Cars dataset
cars = pd.read_csv('data/cleaned_cars.csv')
X = cars.drop('price', axis=1)
y = cars['price']
# st.write(X.head())
# st.write(y.head())

# Sidebar
st.sidebar.header('Specify Input Features')

def user_input_features():
    BRAND = st.sidebar.selectbox()
    MODEL = st.sidebar.selectbox()
    TYPE = st.sidebar.selectbox()
    YEAR = st.sidebar.slider()
    KMS = st.sidebar.slider()
    CITY = st.sidebar.selectbox()
    GEARBOX = st.sidebar.selectbox()
    DOORS = st.sidebar.selectbox()
    SEATS = st.sidebar.selectbox()
    POWER = st.sidebar.slider()
    COLOR = st.sidebar.selectbox()
    CO2 = st.sidebar.slider()
    FUEL = st.sidebar.selectbox()
    WARRANTY = st.sidebar.selectbox()
    DEALER = st.sidebar.selectbox()
    CHASSIS = st.sidebar.selectbox()
    HEIGHT = st.sidebar.slider()
    LENGTH = st.sidebar.slider()
    WIDTH = st.sidebar.slider()
    TRUNK = st.sidebar.slider()
    SPEED = st.sidebar.slider()
    CONS = st.sidebar.slider()
    WEIGHT = st.sidebar.slider()
    TANK = st.sidebar.slider()
    ACC = st.sidebar.slider()
    
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
df = user_input_features()
st.subheader('User Inputs: Technical specs')
st.write(df)
st.write('---')

# Feature Engineering


# Column Transformation: QuantileTransformer


# Load in model
cb_model = decompress_pickle("models/cb_model.pbz2")

# Apply model to predict price
prediction = cb_model.predict(df)
st.write('Prediction of second-hand car price:')
st.write(prediction)
st.write('---')

# Explaining model's ouput predictions using SHAP API
explainer = shap.TreeExplainer(cb_model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance to the prediction')

plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (bar)')
shap.summary_plot(shap_values, X, plot_type='bar')
st.pyplot(bbox_inches='tight')

st.write('---')

# Final reference to the project
st.subheader('References')
st.write("""
For further details regarding this project, please refer to its [repo on GitHub](https://github.com/caresppen/UsedCarsAppraiser).
Here, you will be able to find all the scripts and notebooks used in dataset creation, analysis, visualizations and modeling. You can also download the models used in this app and use them for any other aims.
""")