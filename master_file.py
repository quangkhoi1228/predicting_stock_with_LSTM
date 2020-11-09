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

# -------------------------------declare variable-----------
# symbol name
symbol = 'STB'
# trading view get data url
trading_view_data_url = 'https://iboard.ssi.com.vn/dchart/api/history'
# save csv data path
data_path = f'{os.getcwd()}/data/'
csv_file_name = f'{symbol}.csv'
data_csv_path = f'{data_path}{csv_file_name}'

#epochs: number of train model
epochs = 5
#moving_average_range: range of moving averages from 10 to (n+1)*10
moving_average_range = 5
#predict_delay_session_list: to caculate t+n accuracy
predict_delay_session_list = [1, 3, 5, 10, 20]

# save json result path
json_path = f'{os.getcwd()}/web/json/{symbol}.json'

# -------------------------------get data symbol-----------
# create get data option input json
get_data_options = {
    'symbol': symbol,
    'data_path': data_csv_path,
    'trading_view_data_url': trading_view_data_url
}
# get data symbol
get_data_result = data_from_tradingview(get_data_options)
if get_data_result == True:
    # -------------------------------train,valuation model-------
    file_name = csv_file_name

    model_handle_options = {
        'epochs_range': epochs,
        'moving_average_range': moving_average_range,
        'predict_delay_session_list': predict_delay_session_list,
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

else:
    print('the data is blank or the data is not enough')

