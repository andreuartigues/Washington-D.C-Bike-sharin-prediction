import streamlit as st
from pandas import read_csv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Consumer Behaviour", page_icon="🧑", layout="wide")

from my_streamlit_functions import dataset_clean_complete, generate_stacked_area_chart,temperature_by_month_year, percentage_dist,plot_humidity_and_wind,plot_casuals_weathersit

@st.cache_data
def st_read_csv():
     return read_csv('bike_sharing_hourly.csv')
csv = st_read_csv()

@st.cache_data
def st_dataset_clean():
    return dataset_clean_complete(csv)
data_complete = st_dataset_clean()


st.markdown("# Insights about the Bike Sharing Service")
st.sidebar.header("Time Range Control")

##Month and year selection
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
years = [2011, 2012]
year = st.sidebar.selectbox("Year", years,index=0)
month = st.sidebar.selectbox("Month", month_names, index=0)



st.markdown("We present some insights about how the different bikers use the service and how different feautres affect the number of users.")

st.markdown("### Distribution of Casual and Registered Users")
@st.cache_data
def user_histogram(data):
    fig_histogram = make_subplots(rows=1, cols=2)
    fig_histogram.add_trace(go.Histogram(x=data['casual'], name='Casuals'), row=1, col=1)
    fig_histogram.add_trace(go.Histogram(x=data['registered'], name='Registered'), row=1, col=2)
    st.plotly_chart(fig_histogram, use_container_width=True)
user_histogram(data_complete)

##Percentage distribution
st.markdown("### Percentage of Casual Bikers over time")
fig_percen = percentage_dist(data_complete)
st.plotly_chart(fig_percen, use_container_width=True)

st.markdown("---")
#Time data
st.markdown(f"<h2 style='text-align: center;'>Time Data for {month}, {year}</h2>", unsafe_allow_html=True)

#Stacked area chart
st.markdown("### Hourly evolution of Casual and Registered Users over Time")
fig = generate_stacked_area_chart(data_complete, month_names.index(month) + 1,  year)
st.plotly_chart(fig, use_container_width=True)


#Temp atemp
st.markdown("### Temperature and Feeling Temperature Correlation")
fig_temp= temperature_by_month_year(data_complete, month_names.index(month) + 1,  years.index(year))
st.plotly_chart(fig_temp, use_container_width=True)


##Humidity
st.markdown("### Total Usage depending on Humidity Levels and Windspeeds")
fig_humidity_wind = plot_humidity_and_wind(data_complete, month_names.index(month) + 1,  years.index(year))
st.plotly_chart(fig_humidity_wind, use_container_width=True)


#Boxplot
st.markdown("### Casual users depending on the Weather Situation")
fig_boxplot = plot_casuals_weathersit(data_complete, month_names.index(month) + 1, years.index(year))
st.plotly_chart(fig_boxplot, use_container_width=True)
