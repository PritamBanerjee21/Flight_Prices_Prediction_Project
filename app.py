# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import warnings
import os
from feature_engine.encoding import RareLabelEncoder
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,PowerTransformer,FunctionTransformer,OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from feature_engine.datetime import DatetimeFeatures
from feature_engine.outliers import Winsorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from feature_engine.selection import SelectBySingleFeaturePerformance
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import streamlit as st

# Setting Configurations
pd.set_option("display.max_columns",None)
sklearn.set_config(transform_output="pandas")
warnings.filterwarnings("ignore")

# Feature Engineering
airline_transformer = Pipeline(steps = [
    ('group',RareLabelEncoder(tol = 0.1,replace_with='other',n_categories=2)),
    ('ohe',OneHotEncoder(sparse_output=False,handle_unknown='ignore',drop='first'))
])

date_transformer = Pipeline(steps=[
    ('extraction',DatetimeFeatures(features_to_extract=['month','week','day_of_week','day_of_year'],yearfirst=True,format='mixed')),
    ('scaler',MinMaxScaler())
])

loc_pipe1 = Pipeline(steps=[
    ('grouping',RareLabelEncoder(tol=0.1,replace_with='other',n_categories=2)),
    ('encoding',OneHotEncoder(sparse_output=False,handle_unknown='ignore',drop='first'))
])
def is_south(data):
    south_cities = ['Banglore','Chennai','Cochin','Hyderabad']
    columns = data.columns.to_list()
    return (
        data
        .assign(**{
            f'{col}_is_south' : data.loc[:,col].isin(south_cities).astype(int)
            for col in columns
        })
        .drop(columns = columns)
    )
loc_transformer = FeatureUnion(transformer_list=[
    ('pipeline',loc_pipe1),
    ('function',FunctionTransformer(func=is_south))
])

time_pipe1 = Pipeline(steps=[
    ('extraction',DatetimeFeatures(features_to_extract=['hour','minute'])),
    ('scaing',MinMaxScaler())
])
def day_time(data):
    columns = data.columns.to_list()
    tem_data = data.assign(**{
        col : pd.to_datetime(data.loc[:,col]).dt.hour
        for col in columns
    })
    return(
        data
        .assign(**{
            f'{col}_day_time' : np.select(
                [
                    tem_data.loc[:,col].between(4,12,inclusive='left'),
                    tem_data.loc[:,col].between(12,21,inclusive='left')
                ],
                ['morning','afternoon'],
                default='night'
            )
            for col in columns
        })
        .drop(columns=columns)
    )
time_pipe2 = Pipeline(steps=[
    ('creation',FunctionTransformer(func=day_time)),
    ('encoding',OneHotEncoder(sparse_output=False,handle_unknown='ignore',drop='first'))
])
time_transformer = FeatureUnion(transformer_list=[
    ('pipe1',time_pipe1),
    ('pipe2',time_pipe2)
])

def dur_cat(data,start = 0, short = 180, med = 540, high=5000):
    return (
        data
        .assign(duration_category = np.select(
            [
                data.duration.between(start,short,inclusive = 'left'),
                data.duration.between(short,med,inclusive = 'left')
            ],
            ['short','medium'],
        default='long'
        )
                  )
        .drop(columns = 'duration')
    )
def is_over_1000(data):
    return (
        data
        .assign(over_1000 = data.duration.ge(1000).astype(int))
        .drop(columns = 'duration')
    )
dur_pipe1 = Pipeline(steps=[
    ('function',FunctionTransformer(func=dur_cat)),
    ('encoding',OrdinalEncoder(categories=[['short','medium','long']]))
])
dur_union = FeatureUnion(transformer_list=[
    ('pipe1',dur_pipe1),
    ('func',FunctionTransformer(func=is_over_1000)),
    ('scaling',StandardScaler())
])
duration_transformer = Pipeline(steps=[
    ('outlier',Winsorizer(capping_method='iqr',fold=1.5)),                                # handling outliers by capping them using IQR
    ('union',dur_union)
])

def direct_flight(data):
    return (
        data
        .assign(is_direct_flight = data.total_stops.eq(0).astype(int))
    )

stop_transformer = FunctionTransformer(func=direct_flight)

info_pipe1 = Pipeline(steps=[
    ('grouping',RareLabelEncoder(tol=0.1,n_categories=2,replace_with='other')),
    ('encoding',OneHotEncoder(handle_unknown='ignore',sparse_output=False,drop='first'))
])
def info(data):
    return (
        data
        .assign(additional_info = data.additional_info.ne('No Info').astype(int))
    )
info_transformer = FeatureUnion(transformer_list=[
    ('pipe',info_pipe1),
    ('func',FunctionTransformer(func=info))
])



transformer = ColumnTransformer(transformers=[
    ('air',airline_transformer,['airline']),
    ('journey',date_transformer,['date_of_journey']),
    ('location',loc_transformer,['source','destination']),
    ('time',time_transformer,['dep_time','arrival_time']),
    ('flight',duration_transformer,['duration']),
    ('break',stop_transformer,['total_stops']),
    ('travel',info_transformer,['additional_info'])
],remainder='passthrough')


estimator = RandomForestRegressor(n_estimators=40,max_depth=5,n_jobs=-1,random_state=42)

selector = SelectBySingleFeaturePerformance(estimator=estimator,scoring='r2',cv = 10,threshold=0.05)


features = Pipeline(steps=[
    ('column transform',transformer),
    ('selection',selector)
])

# Reading Training Data
train = pd.read_csv("train.csv")
X_train = train.drop(['price'],axis=1)
y_train = train.price

# Fit and Save the preprocessor
features.fit(X_train,y_train)
joblib.dump(features,"preprocessor.joblib")

# web application
st.set_page_config(
	page_title="Flights Prices Prediction",
	page_icon="✈️",
	layout="wide"
)
st.title("Flights Prices Predictor")

# user inputs
airline = st.selectbox(
	"Airline:",
	options=X_train.airline.unique()
)

doj = st.date_input("Date of Journey:")

source = st.selectbox(
	"Source",
	options=X_train.source.unique()
)

destination = st.selectbox(
	"Destination",
	options=X_train.destination.unique()
)

dep_time = st.time_input("Departure Time:")

arrival_time = st.time_input("Arrival Time:")

duration = st.number_input(
	"Duration (mins):",
	step=1
)

total_stops = st.number_input(
	"Total Stops:",
	step=1,
	min_value=0
)

additional_info = st.selectbox(
	"Additional Info:",
	options=X_train.additional_info.unique()
)

x_new = pd.DataFrame(dict(
	airline=[airline],
	date_of_journey=[doj],
	source=[source],
	destination=[destination],
	dep_time=[dep_time],
	arrival_time=[arrival_time],
	duration=[duration],
	total_stops=[total_stops],
	additional_info=[additional_info]
)).astype({
	col: "str"
	for col in ["date_of_journey", "dep_time", "arrival_time"]
})


if st.button("Predict"):
        saved_preprocessor = joblib.load("preprocessor.joblib")
        x_new_pre = saved_preprocessor.transform(x_new)
        model = joblib.load("XGBoost.joblib")

        # with open("xgboost-model", "rb") as f:
        # 	model = pickle.load(f)
        # x_new_xgb = xgb.DMatrix(x_new_pre)
        pred = model.predict(x_new_pre)

        st.info(f"The predicted price is {pred} INR")