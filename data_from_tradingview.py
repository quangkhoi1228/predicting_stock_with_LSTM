import requests
import json
import pandas as pd
import os
import datetime


def data_from_tradingview(get_data_options):
    symbol = get_data_options['symbol']
    data_path = get_data_options['data_path']
    trading_view_data_url = get_data_options['trading_view_data_url']

    data_json = {
        'date': [],
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': [],
    }
    now = datetime.datetime.now()
    current_year = now.year
    for year in range(2000, current_year):

        begin_date = int(datetime.datetime(year, 1, 1).timestamp())
        end_date = int(datetime.datetime(year, 12, 31).timestamp())

        print(f'{year}')
        response = requests.get(
            f'{trading_view_data_url}?symbol={symbol}&resolution=D&from={begin_date}&to={end_date}')
        response_data = json.loads(response.text)
        data_json['date'].extend(response_data['t'])
        data_json['open'].extend(response_data['o'])
        data_json['high'].extend(response_data['h'])
        data_json['low'].extend(response_data['l'])
        data_json['close'].extend(response_data['c'])
        data_json['volume'].extend(response_data['v'])

    df_file = pd.DataFrame(data_json)
    df_file.to_csv(data_path, index=False)
