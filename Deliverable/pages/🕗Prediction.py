import streamlit as st
import joblib 
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from my_streamlit_functions import dataset_clean_short,dataset_clean_complete, transform, my_metrics, my_error_plot, plot_radar_chart

st.set_page_config(page_title="Predictive Model", page_icon="💻", layout="wide")
st.title("Predictive Model and Discussions")

@st.cache_data
def st_read_csv():
     return read_csv('bike_sharing_hourly.csv')
csv = st_read_csv()

@st.cache_data
def st_dataset_clean():
    return dataset_clean_complete(csv)
data_complete = st_dataset_clean()

@st.cache_data
def st_dataset_clean_shorted():
    return dataset_clean_short(data_complete)
data= st_dataset_clean_shorted()

@st.cache_data
def data_split(data):
    data_casual= data.drop(columns=['registered','cnt'], axis=1)
    data_regular=data.drop(columns=['casual','cnt'], axis=1)
    return data_casual, data_regular
data_casual, data_regular= data_split(data)


@st.cache_data
def model_tuning_casual(data_casual):
    X = data.drop(columns=['casual','holiday'], axis=1)
    y = data["casual"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
    
    preprocessing=transform(X)
    
    return X_train, y_train, X_test, y_test,preprocessing
X_train_c, y_train_c, X_test_c, y_test_c,preprocessing = model_tuning_casual(data_casual)

@st.cache_data
def model_tuning_regular(data):
    X = data.drop(columns=['registered','holiday'], axis=1)
    y = data["registered"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
    
    preprocessing=transform(X)
    
    return X_train, y_train, X_test, y_test,preprocessing

X_train_r, y_train_r, X_test_r, y_test_r,preprocessing = model_tuning_regular(data_regular)


st.markdown("### Linear Regression residuals")

#@st.cache_data
def various_casual_linear():
    model = joblib.load('models/model_casual_linear.pkl')
    error_plot=my_error_plot(model, X_test_c, y_test_c)
    st.plotly_chart(error_plot, use_container_width=True)
    return model

model_linear=various_casual_linear()

st.markdown("#### LightGBRegressor residuals: Casual Users")

#@st.cache_data
def various_casual_light():
    model = joblib.load('models/model_casual_lightgmr.pkl')
    error_plot=my_error_plot(model, X_test_c, y_test_c)
    return model, st.plotly_chart(error_plot, use_container_width=True)

model_casual, error_casual=various_casual_light()


st.markdown("### LightGBRegressor residuals: Registered Users")

#@st.cache_data
def various_regular():
    model = joblib.load('models/model_lightgmr.pkl')
    error_plot=my_error_plot(model, X_test_r, y_test_r)
    return model, st.plotly_chart(error_plot, use_container_width=True)

model_regular, error_regular =various_regular()


st.sidebar.header("Independent Features")


dict = {
    'Season 🍂': ['Winter', 'Spring', 'Summer', 'Fall'],
    'Month 📅': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'Weekday 📆': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'Working Day 🧑‍💼': ['No', 'Yes'],
    'Weather Situation ☔': ['Clear or Partly cloudy', 'Mist / Cloudy', '(Light) Snow/Rain or Thunderstorm']
}

season_month = {
    'Winter': [1, 2, 3, 12],
    'Spring': [3, 4, 5, 6],
    'Summer': [6, 7, 8, 9],
    'Fall': [9, 10, 11, 12]
}

weekday_conversion = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}

workingday_conversion = {   
    0: 'No',
    1: 'Yes'
}

weather_situations = {
    1: 'Clear or Partly cloudy',
    2: 'Mist / Cloudy',
    3: '(Light) Snow/Rain or Thunderstorm '
}

season = st.sidebar.selectbox("Season 🍂", options=dict['Season 🍂'])
months_in_season = season_month[season]
month_options = [month_name for month_num, month_name in [
    (1, 'January'), (2, 'February'), (3, 'March'), (4, 'April'),
    (5, 'May'), (6, 'June'), (7, 'July'), (8, 'August'),
    (9, 'September'), (10, 'October'), (11, 'November'), (12, 'December')
] if month_num in months_in_season]

selected_month = st.sidebar.selectbox("Month 📅", options=month_options)


# Define default values
default_hour = np.random.randint(8, 13)
default_temp = np.random.uniform(10, 20)
default_humidity = np.random.uniform(40, 60)
default_windspeed = np.random.uniform(10, 20)

# Check if session state exists, if not create one
if 'slider_values' not in st.session_state:
    st.session_state.slider_values = {
        'Hour': default_hour,
        'Temperature': default_temp,
        'Humidity': default_humidity,
        'Windspeed': default_windspeed
    }

selected_values = {
    'Season': dict['Season 🍂'].index(season) + 1,  
    'Month': dict['Month 📅'].index(selected_month) + 1, 
    'Hour': st.sidebar.slider("Hour ⏱️", min_value=data['hr'].min(), max_value=data['hr'].max(), value=st.session_state.slider_values['Hour']),
    'Weekday': list(weekday_conversion.keys())[list(weekday_conversion.values()).index(st.sidebar.selectbox("Weekday 📆", options=dict['Weekday 📆']))],  
    'Working Day': list(workingday_conversion.keys())[list(workingday_conversion.values()).index(st.sidebar.selectbox("Working Day 🧑‍💼", options=dict['Working Day 🧑‍💼']))], 
    'Weather Situation': list(weather_situations.keys())[list(weather_situations.values()).index(st.sidebar.selectbox("Weather Situation ☔", options=dict['Weather Situation ☔']))],  
    'Temperature': st.sidebar.slider("Temperature 🌡️", min_value=data['temp'].min(), max_value=data['temp'].max(), value=st.session_state.slider_values['Temperature']),
    'Humidity': st.sidebar.slider("Humidity 💧", min_value=data['hum'].min(), max_value=data['hum'].max(), value=st.session_state.slider_values['Humidity']),
    'Windspeed': st.sidebar.slider("Windspeed 🌬️", min_value=data['windspeed'].min(), max_value=data['windspeed'].max(), value=st.session_state.slider_values['Windspeed'])
}

# Update session state values
st.session_state.slider_values['Hour'] = selected_values['Hour']
st.session_state.slider_values['Temperature'] = selected_values['Temperature']
st.session_state.slider_values['Humidity'] = selected_values['Humidity']
st.session_state.slider_values['Windspeed'] = selected_values['Windspeed']


selected_values['season'] = selected_values.pop('Season')
selected_values['mnth'] = selected_values.pop('Month')
selected_values['hr']= selected_values.pop('Hour')
selected_values['weekday'] = selected_values.pop('Weekday')
selected_values['workingday'] = selected_values.pop('Working Day')
selected_values['weathersit'] = selected_values.pop('Weather Situation')
selected_values['temp']= selected_values.pop('Temperature')
selected_values['hum']= selected_values.pop('Humidity')
selected_values['windspeed']= selected_values.pop('Windspeed')


new_data = pd.DataFrame(selected_values, index=[0]).astype({
        'season': 'object',
        'weekday': 'object',
        'workingday': 'object',
        'weathersit': 'object',
        'mnth': 'object'
    })

predictions_casual = model_casual.predict(new_data).astype(int)
predictions_regular = model_regular.predict(new_data).astype(int)

error_casual = predictions_casual*0.25
error_regular = predictions_regular*0.23

predictions_casual = str(predictions_casual[0])
predictions_regular = str(predictions_regular[0])

st.markdown("### Radar Chart of Feature Importance")
categories = ['Hour', 'Temperature', 'Humidity', 'Windspeed', 'Weekday', 'Weather Situation', 'Month', 'Workingday', 'Season']
values = [1917, 1317, 1165, 730, 140, 90, 70, 267, 83]
st.plotly_chart(plot_radar_chart(categories, values),use_container_width=True)


st.markdown("---")

st.markdown(f"<h1 style='text-align: center;'>Predictions <h1>", unsafe_allow_html=True)

col1, col2=st.columns(2)
with col1:
    st.write(f"#### Predicted number of casual bikes: {predictions_casual}")

with col2:
    st.write(f"#### Predicted number of registered bikes: {predictions_regular}")
    
st.markdown("---")