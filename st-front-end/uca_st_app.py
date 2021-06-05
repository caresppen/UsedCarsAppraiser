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

st.header("Used Cars Appraisser")
st.write("""
This application is based on a ML Random Forest algorithm. The model was trained using a dataset of 55,366 real second-hand cars from [coches.com](https://www.coches.com/). It can predict prices of used cars up to 100,000€
""")
st.write('---')

st.sidebar.header('User Input Features')

def user_input_features():
    BRAND = st.sidebar.selectbox()
    MODEL = st.sidebar.selectbox()
    TYPE = st.sidebar.selectbox()
    YEAR = st.sidebar.slider()
    KMS = st.sidebar.slider()
    CITY = st.sidebar.selectbox()
    GEARBOX = st.sidebar.selectbox()
    DOORS = st.sidebar.selectbox()
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
    
    data = {'brand',
            'model',
            'type',
            'year'
            'city',
            'gearbox',
            'doors',
           }
    