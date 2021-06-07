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
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer

st.header("Used Cars **App**raisser")
st.write("""
This application is based on a ML Random Forest algorithm. The model was trained using a dataset of 55,326 real second-hand cars from [coches.com](https://www.coches.com/). It can predict prices of used cars up to 100,000€
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
    BRAND = st.sidebar.selectbox('Brand', X.brand.unique())
    MODEL = st.sidebar.selectbox('Model', X[X.brand == BRAND].model.unique())
    TYPE = st.sidebar.selectbox('Type', X.type.unique())
    CITY = st.sidebar.selectbox('City', X.city.unique())
    GEARBOX = st.sidebar.selectbox('Gearbox', X.gearbox.unique())
    COLOR = st.sidebar.selectbox('Color', X.color.unique())
    FUEL = st.sidebar.selectbox('Fuel', X.fuel_type.unique())
    WARRANTY = st.sidebar.selectbox('Warranty', X.warranty.unique())
    DEALER = st.sidebar.selectbox('Dealer', X.dealer.unique())
    CHASSIS = st.sidebar.selectbox('Chassis', X.chassis.unique())
    YEAR = st.sidebar.slider('Year', X.year.min(), X.year.max(), int(round(X.year.mean(),0)))
    KMS = st.sidebar.slider('Kms', X.kms.min(), X.kms.max(), int(round(X.kms.mean(),0)))
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
df = user_input_features()
st.subheader('User Inputs: Technical specs')
st.write(df)
st.write('---')

### Feature Engineering
ohe_cols = ['gearbox', 'fuel_type', 'warranty', 'dealer', 'doors']

# OHE
ohe = OneHotEncoder(categories='auto')
ohe.fit_transform(X[ohe_cols]).toarray()
feature_arr = ohe.transform(df[ohe_cols]).toarray()
feature_labels = ohe.categories_

# Using a dictionary to produce all the new OHE columns
feature_cols = []
for k, v in dict(zip(ohe_cols, feature_labels)).items():
    for i in v:
        el = k + '_' + str(i)
        feature_cols.append(el)

ohe_features = pd.DataFrame(feature_arr, columns=feature_cols)
df = pd.concat([df, ohe_features], axis=1).drop(ohe_cols, axis=1)

# Target Encoding
cat_cols = df.select_dtypes(exclude=["number"]).columns
cols_encoded = list(map(lambda c: c + '_encoded', cat_cols))

t_encoder = TargetEncoder()
X_enc = t_encoder.fit_transform(X[cat_cols], y)
df[cols_encoded] = t_encoder.transform(df[cat_cols])
df = df.drop(cat_cols, axis=1)

# Column Transformation: QuantileTransformer
qt = QuantileTransformer(n_quantiles=500,
                         output_distribution='normal',
                         random_state=33)

qt.fit_transform(X_enc) # y: How to apply Target Encoding to new data after trained model???
df = qt.transform(df)

st.write(df)

# Load in model
cb_model = decompress_pickle("notebooks/models/cb_model.pbz2")

# Apply model to predict price
prediction = cb_model.predict(df)
st.write('Prediction of second-hand car price:')
st.write(prediction)
st.write('---')

# Explaining model's ouput predictions using SHAP API
X = cars.drop(['title', 'price'], axis=1)
y = cars['price']

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
Created by Carlos Espejo Peña.
""")