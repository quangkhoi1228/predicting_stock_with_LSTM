import requests
import json
import pandas as pd
import os
import datetime
import urllib
from datetime import datetime
import matplotlib.pyplot as plt
from util import csv_to_dataset, history_points
import keras
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
import json
import webbrowser


# import get data function
from data_from_tradingview import data_from_tradingview
# import model handle function
from model_handle_util import train_predict, statistic_result
# import view result function
from view_all_symbol import view_result


# symbol name
symbol = 'SSI'
# trading view get data url
trading_view_data_url = 'https://iboard.ssi.com.vn/dchart/api/history'
# save csv data path
data_path = f'{os.getcwd()}/data/'
csv_file_name = f'{symbol}.csv'
data_csv_path = f'{data_path}{csv_file_name}'
# save json result path
json_path = f'{os.getcwd()}/web/json/{symbol}.json'

# -------------------------------declare variable-----------
# create get data option input json
get_data_options = {
    'symbol': symbol,
    'data_path': data_csv_path,
    'trading_view_data_url': trading_view_data_url
}
# get data symbol
data_from_tradingview(get_data_options)

# -------------------------------train,valuation model-------
file_name = csv_file_name

model_handle_options = {
    'epochs_range': 1,
    'moving_average_range': 1,
    'predict_delay_session_list': [1, 3, 5, 10, 20],
    'file_name': file_name,
    'data_path': data_path,
    'data_csv_path': data_csv_path
}

train_predict(model_handle_options)
list_statistic = statistic_result(model_handle_options)


#-------------------------------view result as web page------
view_result_options = {
    'json_data_file_path': json_path,
    'symbol': symbol

}
view_result(view_result_options)
