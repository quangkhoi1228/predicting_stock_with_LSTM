import pandas as pd
import os
import numpy as np


def remove_nan_value(data_frame):
    lst = data_frame['volume']
    last_index = 0
    for index, item in enumerate(lst):
        # print(f'item {item}')
        if(index > 0):
            if np.isnan(np.sum(item)):
                last_index = index

    for key in data_frame.keys():
        data_frame[key] = data_frame[key][(last_index+1):]

    return data_frame

def base_data_excel_handle():
    df = pd.read_excel(f'{os.getcwd()}/base_data/data3.xlsx', 1)

    # crop date column
    columns = df.columns
    date_column_name = columns[0]
    date_column = df[date_column_name]
    del df[date_column_name]
    while len(df.columns) > 0:
        print(len(df.columns))
        data_frame = {
            'date': date_column.tolist()
        }
        current_header = df.columns[:5]
        file_name = current_header[0].replace(
            ' ', '_').replace('_VM_Equity', '').replace('_VM_Equity', '')

        for index, column_item in enumerate(current_header):
            data_column_item = df[column_item]
            data_frame[f'index_{index}'] = data_column_item.tolist()
            del df[column_item]

        data_frame['date'] = data_frame.pop('date')
        data_frame['open'] = data_frame.pop('index_0')
        data_frame['high'] = data_frame.pop('index_1')
        data_frame['low'] = data_frame.pop('index_2')
        data_frame['close'] = data_frame.pop('index_3')
        data_frame['volume'] = data_frame.pop('index_4')
        #remove nan value
        data_frame = remove_nan_value(data_frame)

        df_file = pd.DataFrame(data_frame)

        df_file.to_csv(f'{os.getcwd()}/data/VN30/{file_name}.csv', index=False)


base_data_excel_handle()
