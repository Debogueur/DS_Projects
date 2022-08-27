# 1 --- first and foremost, we import the necessary libraries
import pandas as pd
import streamlit as st
import numpy as np
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from tensorflow import keras
from datetime import date,timedelta

from statsmodels.tsa.arima.model import ARIMAResults
import pickle
#import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages
#######################################

def load_Model():
    #Load Model
    fm = ARIMAResults.load("finalized_model.pkl")
    return fm

def predict(end_year):
    fm =load_Model()
    pred = fm.predict(start = '1970-01-01', end = end_year , dynamic= False).to_frame('CO2')
    past_data = np.exp(pred).iloc[0:45,:]
    result = np.exp(pred)[44:]
    result.index = result.index.date
    st.write(result)
    return result,past_data

def plot_graph(past_data,pred,periods_input):

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.style.use('dark_background')
    fig,ax = plt.subplots(figsize=(8,5))
    ax.plot(past_data,color = 'blue',label='Past 45 years data')
    ax.plot(pred,color = 'orange',label='Next '+str(periods_input)+' Years Data')
    ax.legend()
    st.pyplot(fig,clear_figure=False)


def main():

    st.title("Forecast Co2 Levels")
    periods_input = st.number_input('How many years forecast do you want from 2014-01-01?',
    min_value = 1, max_value = 100)
    last_predicted_year = 2014
    end_year = str(last_predicted_year + periods_input)+'-01-01'
    if st.button('Predict'):
        st.write("CO2 Emission From 2014-01-01 to "+end_year)
        future_data,past_data = predict(end_year)
        plot = plot_graph(past_data,future_data,periods_input)
        

if __name__ == '__main__':
    main()






