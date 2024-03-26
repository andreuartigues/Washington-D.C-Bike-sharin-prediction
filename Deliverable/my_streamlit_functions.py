#------------------------------#
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
#------------------------------#
from scipy.stats import uniform, randint
from sklearn.preprocessing import OneHotEncoder
#------------------------------#
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
#------------------------------#
import warnings
import pandas as pd
warnings.filterwarnings('ignore')


def dataset_clean_complete(data):
    data = data.astype({
        'season': 'object',
        'yr': 'object',
        'holiday': 'object',
        'weekday': 'object',
        'workingday': 'object',
        'weathersit': 'object',
        'mnth': 'object'
    })
    data.dteday = pd.to_datetime(data.dteday)
    data['dteday'] = data['dteday'] + pd.to_timedelta(data['hr'], unit='h')
    data.loc[data['weathersit'] == 4, 'weathersit'] = 3
    columns = ['temp', 'atemp', 'hum', 'windspeed']  
    max_values = [41, 50, 100,67] 
    data[columns] = data[columns].mul(max_values)
    return data

def dataset_clean_short(data):
    data.drop(columns=['instant','dteday','yr','atemp'], axis=1,inplace=True)
    return data

month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
def generate_stacked_area_chart(data,month, year):
    filtered_data = data[(data['dteday'].dt.month == month) & (data['dteday'].dt.year == year)]
    fig = px.area(filtered_data, x='dteday', y=['casual', 'registered'])
    fig.update_layout(xaxis_title='',yaxis_title=None)
    
    return fig
    
    
def percentage_dist(data):
    data['year'] = data['dteday'].dt.year
    data['month'] = data['dteday'].dt.month

    monthly_data = data.groupby(['year', 'month']).agg(total_users=('cnt', 'sum'),
                                                    casual_users=('casual', 'sum'))

    monthly_data['casual_percentage'] = (monthly_data['casual_users'] / monthly_data['total_users']) * 100

    monthly_data.reset_index(inplace=True)

    fig = px.line(monthly_data, x='month', y='casual_percentage', color='year', line_group='year',
                labels={'casual_percentage': 'Casual Bikers (%)', 'month': 'Month', 'year': ''}) 
    fig.update_traces(mode='markers+lines')
    fig.add_shape(type='line',
                x0=0,
                x1=13,
                y0=19.88,
                y1=19.88,
                line=dict(color='blue', width=1.5, dash='dash'))
    fig.add_shape(type='line',
                x0=0,
                x1=13,
                y0=18.18,
                y1=18.18,
                line=dict(color='darkcyan', width=1.5, dash='dash'))
    data.drop(columns=['year','month'], axis=1,inplace=True)
    return fig



def plot_casuals_weathersit(data, month, year):
    filtered_data = data[(data['mnth'] == month) & (data['yr'] == year)]
    
    fig = go.Figure()
    
    weather_mapping = {1: 'Clear or Partly cloudy⛅', 2: 'Mist / Cloudy🌦️', 3: '(Light) Snow/Rain<br> or Thunderstorm ⛈️'}
    filtered_data['weathersit'] = filtered_data['weathersit'].map(weather_mapping)
    
    for category in filtered_data['weathersit'].unique():
        fig.add_trace(go.Box(
            x=filtered_data[filtered_data['weathersit'] == category]['weathersit'],
            y=filtered_data[filtered_data['weathersit'] == category]['cnt'],
            name=f"{category}"
        ))

    fig.update_layout(
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig



def temperature_by_month_year(data,month, year):
    filtered_data = data[(data['mnth'] == month) & (data['yr'] == year)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data['dteday'], y=filtered_data['temp'], name='Temperature'))
    fig.add_trace(go.Scatter(x=filtered_data['dteday'], y=filtered_data['atemp'], name='Feeling Temperature'))
    
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Temperature (Celsius)')
    
    return fig

def plot_humidity_and_wind(data, month, year):
    filtered_data = data[(data['mnth'] == month) & (data['yr'] == year)]
    
    fig = make_subplots(rows=2, cols=1)
    
    fig.add_trace(go.Scatter(x=filtered_data['hum'], y=filtered_data['cnt'], mode='markers'), row=1, col=1)
    fig.add_trace(go.Scatter(x=filtered_data['windspeed'], y=filtered_data['cnt'], mode='markers'), row=2, col=1)
    
    fig.update_layout(xaxis_title='Humidity', yaxis_title='', xaxis2_title='Wind', yaxis2_title='',height=600,showlegend=False)
    
    return fig


### Functions for predictions
### Functions for predictions
### Functions for predictions



def transform(X):
    categorical_columns = X.select_dtypes(["O"]).columns
    numerical_columns = X.select_dtypes(["int","float"]).columns
    preprocessing = ColumnTransformer(
    [
        (
            "ohe",
            OneHotEncoder(sparse_output=False),
            categorical_columns
        ),
        (
            "scaler",
            MinMaxScaler(),
            numerical_columns
        )
    ],
    remainder="passthrough"  # this can be "drop", "passthrough", or another Estimator
)
    
    return preprocessing

def my_metrics(model, X_train, y_train, X_test, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    train_metrics = {
        "R2": r2_score(y_train, y_train_pred),
        "RMSE": mean_squared_error(y_train, y_train_pred, squared=False),
        "MAE": mean_absolute_error(y_train, y_train_pred)
    }
    
    test_metrics = {
        "R2": r2_score(y_test, y_test_pred),
        "RMSE": mean_squared_error(y_test, y_test_pred, squared=False),
        "MAE": mean_absolute_error(y_test, y_test_pred)
    }
    
    train_metrics_df = pd.DataFrame(train_metrics, index=["Train"])
    test_metrics_df = pd.DataFrame(test_metrics, index=["Test"])
    
    metrics = pd.concat([train_metrics_df, test_metrics_df])
    
    return metrics
 
def my_error_plot(model, X_test, y_test):   
    y_test_pred = model.predict(X_test)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test.values,
        y=y_test_pred,
        mode="markers",
        name="Samples"
    ))
    
    fig.add_trace(go.Scatter(
        x=y_test.values,
        y=y_test.values,
        mode="lines",
        name="Reference"
    ))
    
    fig.update_layout(
        xaxis_title="True",
        yaxis_title="Predicted",
        template="none"
    )
    
    return fig

#Not used in the end
def feat_importance(model,X_train,preprocessing):
    feature_importance = model.feature_importances_

    X_train_transformed= pd.DataFrame(preprocessing.fit_transform(X_train), columns=preprocessing.get_feature_names_out())
    feature_names = X_train_transformed.columns

    feature_importance_dict = dict(zip(feature_names, feature_importance))
    sorted_feature_importance = dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(sorted_feature_importance.keys()), y=list(sorted_feature_importance.values())))

    fig.update_layout(
        title='Feature Importance Plot',
        xaxis_title='Feature',
        yaxis_title='Feature Importance'
    )
    return fig

def plot_radar_chart(categories, values):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker=dict(
            color='blue'
        )
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values)], 
            ),
        ),
        showlegend=False
    )
    return fig