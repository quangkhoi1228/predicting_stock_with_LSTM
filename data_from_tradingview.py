import requests
import json
import pandas as pd
import os
import datetime


symbol = 'HNX30'

data_json = {
    'date': [],
    'open': [],
    'high': [],
    'low': [],
    'close': [],
    'volume': [],
}
for year in range(2000, 2030):
    begin_date = int(datetime.datetime(year, 1, 1).timestamp())
    end_date = int(datetime.datetime(year, 12, 31).timestamp())

    print(f'{year},{begin_date}, {end_date}')
    response = requests.get(
        f'https://chart.aladin.finance/history?symbol={symbol}&resolution=D&from=964749600&to=2000574566')
    response_data = json.loads(response.text)
    data_json['date'].extend(response_data['t'])
    data_json['open'].extend(response_data['o'])
    data_json['high'].extend(response_data['h'])
    data_json['low'].extend(response_data['l'])
    data_json['close'].extend(response_data['c'])
    data_json['volume'].extend(response_data['v'])
    
df_file = pd.DataFrame(data_json)
df_file.to_csv(f'{os.getcwd()}/data/index/{symbol}.csv', index=False)
