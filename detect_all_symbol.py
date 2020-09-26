import pandas as pd
import os
import numpy as np
import main
import json
import webbrowser
import threading
import random


from model_handle_util import main

data_path = f'{os.getcwd()}/data/index/'
json_path = f'{os.getcwd()}/web/json/index'

# all_file = os.listdir(data_path)

all_file = ['HNX30.csv']
# list_statistic = {}
for file_name in all_file:

    # for i in range(1000):
    #     file_name = random.choice(all_file)

    try:
        symbol = file_name.replace('.csv', '')
        print(symbol)
        json_data_file_path = f'{json_path}{symbol}.json'
        if os.path.isfile(json_data_file_path) == False:
            options = {
                'input_config': {
                    'data_path': data_path,
                    'epochs_range': 100,
                    'moving_average_range': 10,
                    'predict_delay_session_list': [1, 3, 5, 10, 20],
                    'file_name': file_name,
                }
            }

            symbol_statistic = main(options)
        else:
            json_data_file = open(json_data_file_path, "r")
            symbol_statistic = json.loads(json_data_file.read())
            json_data_file.close()
            print(f'{symbol} has detect')

    except NameError:
        print("An exception occurred")
