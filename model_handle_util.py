import urllib
import os
from datetime import datetime
import matplotlib.pyplot as plt
from util import csv_to_dataset, history_points
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
import numpy as np
from keras.models import load_model
import json
import webbrowser


np.random.seed(4)
tf.random.set_seed(4)


def get_info_from_options(options):
    file_name = options.get(
        'file_name') if 'file_name' in options else 'VN_INDEX_daily.csv'
    data_path = options.get(
        'data_path') if 'data_path' in options else f'{os.getcwd()}/data/index/'
    base_name = file_name.replace(".csv", "")
    file_name_path = f'{data_path}{file_name}'
    moving_average = options.get(
        'moving_average') if 'moving_average' in options else 30
    epochs = options.get('epochs') if 'epochs' in options else 10
    start = options.get('start') if 'start' in options else 0
    end = options.get('end') if 'end' in options else -1
    predict_delay_session_list = options.get(
        'predict_delay_session_list') if 'predict_delay_session_list' in options else [1, 3, 5, 10]
    model_base_name = f'{base_name}_{epochs}_{moving_average}'
    model_name = f'{os.getcwd()}/models/{base_name}_{epochs}_{moving_average}.h5'
    result_name = f'{os.getcwd()}/result/{base_name}.json'
    json_path = f'{os.getcwd()}/web/json/{base_name}.json'

    return {
        'file_name': file_name_path,
        'base_name': base_name,
        'moving_average': moving_average,
        'file_name_path': file_name_path,
        'epochs': epochs,
        'model_name': model_name,
        'model_base_name': model_base_name,
        'start': start,
        'end': end,
        'predict_delay_session_list': predict_delay_session_list,
        'result_name': result_name,
        'json_path': json_path
    }


def check_result_already_exist(options):
    info = get_info_from_options(options)
    moving_average = info['moving_average']
    epochs = info['epochs']
    predict_delay_session_list = info['predict_delay_session_list']
    result_name = info['result_name']
    return_value = True

    if os.path.isfile(result_name) == True:
        pre_content = json.loads(open(result_name, "r").read())
        epochs_key = f'epochs_{epochs}'
        if (epochs_key in pre_content) == False:
            return_value = False
        else:
            moving_average_key = f'moving_average_{moving_average}'
            if (moving_average_key in pre_content[epochs_key]) == False:
                return_value = False
            else:
                for delay_session in predict_delay_session_list:
                    delay_session_key = f'delay_session_{delay_session}'
                    if (delay_session_key in pre_content[epochs_key][moving_average_key]) == False:
                        return_value = False
    else:
        return_value = False

    return return_value


def train_model(options):
    info = get_info_from_options(options)
    file_name = info['file_name']
    moving_average = info['moving_average']
    epochs = info['epochs']
    save_name = info['model_name']
    model_base_name = info['model_base_name']

    print(f'{model_base_name}')

    if os.path.isfile(save_name) == True:
        print(f'{model_base_name} trained')
        return
    try:
        ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(
            file_name)
    except():
        print('data parse false')
    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)
    ohlcv_train = ohlcv_histories[:n]
    tech_ind_train = technical_indicators[:n]
    y_train = next_day_open_values[:n]
    ohlcv_test = ohlcv_histories[n:]
    tech_ind_test = technical_indicators[n:]
    y_test = next_day_open_values[n:]
    unscaled_y_test = unscaled_y[n:]
    # model architecture
    # define two sets of inputs
    lstm_input = Input(shape=(history_points, 5), name='lstm_input')

    dense_input = Input(
        shape=(technical_indicators.shape[1],), name='tech_input')
    # the first branch operates on the first input
    x = LSTM(moving_average, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate(
        [lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[lstm_branch.input,
                          technical_indicators_branch.input], outputs=z)
    adam = optimizers.Adam(lr=0.0005)
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=[ohlcv_train, tech_ind_train], y=y_train,
              batch_size=64, epochs=epochs, shuffle=True, validation_split=0.1)
    # evaluation
    y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
    # print('y_test_predicted.shape', y_test_predicted.shape)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    y_predicted = model.predict([ohlcv_histories, technical_indicators])
    y_predicted = y_normaliser.inverse_transform(y_predicted)
    assert unscaled_y_test.shape == y_test_predicted.shape
    real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
    scaled_mse = real_mse / (np.max(unscaled_y_test) -
                             np.min(unscaled_y_test)) * 100
    # print(scaled_mse)
    plt.gcf().set_size_inches(22, 15, forward=True)
    start = 0
    end = -1

    real = plt.plot(unscaled_y_test[start:end], label='real')
    pred = plt.plot(y_test_predicted[start:end], label='predicted')

    plt.legend(['Real', 'Predicted'])
    # plt.show()
    model.save(save_name)


def predict_model(options):
    info = get_info_from_options(options)
    file_name = info['file_name']
    modal_name = info['model_name']
    start = info['start']
    end = info['end']
    model_base_name = info['model_base_name']
    predict_delay_session_list = info['predict_delay_session_list']
    check_result_already_exist_value = check_result_already_exist(options)
    if check_result_already_exist_value == True:
        print(f'{model_base_name} predicted')
        return False
    model = load_model(modal_name)
    ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(
        file_name)
    test_split = 0.9
    n = int(ohlcv_histories.shape[0] * test_split)

    ohlcv_train = ohlcv_histories[:n]
    tech_ind_train = technical_indicators[:n]
    y_train = next_day_open_values[:n]

    ohlcv_test = ohlcv_histories[n:]
    tech_ind_test = technical_indicators[n:]
    y_test = next_day_open_values[n:]

    unscaled_y_test = unscaled_y[n:]
    y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    buys = []
    sells = []
    thresh = 0.1

    # start = 0
    # end = -1
    x = 0

    todayPriceList = []
    predictedPriceList = []
    length = len(ohlcv_test[start: end])
    for ohlcv, ind in zip(ohlcv_test[start: end], tech_ind_test[start: end]):
        normalised_price_today = ohlcv[-1][0]
        normalised_price_today = np.array([[normalised_price_today]])
        price_today = y_normaliser.inverse_transform(normalised_price_today)
        ohlcv_predict = np.array(ohlcv, ndmin=3)  # fix dimension
        ind_predict = np.array(ind, ndmin=2)  # fix dimension
        predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(
            model.predict([ohlcv_predict, ind_predict])))
        todayPriceList.append(price_today[0][0])
        predictedPriceList.append(float(predicted_price_tomorrow))
        delta = predicted_price_tomorrow - price_today
        if delta > thresh:
            buys.append((x, price_today[0][0]))
        elif delta < -thresh:
            sells.append((x, price_today[0][0]))
        x += 1
        # print(x, '/', length)

    # print(f"buys: {len(buys)}")
    # print(f"sells: {len(sells)}")

    # def compute_earnings(buys_, sells_):
    #     purchase_amt = 10
    #     stock = 0
    #     balance = 0
    #     while len(buys_) > 0 and len(sells_) > 0:
    #         if buys_[0][0] < sells_[0][0]:
    #             # time to buy $10 worth of stock
    #             balance -= purchase_amt
    #             stock += purchase_amt / buys_[0][1]
    #             buys_.pop(0)
    #         else:
    #             # time to sell all of our stock
    #             balance += stock * sells_[0][1]
    #             stock = 0
    #             sells_.pop(0)
    #     print(f"earnings: ${balance}")

    # we create new lists so we dont modify the original
    # compute_earnings([b for b in buys], [s for s in sells])

    # plt.gcf().set_size_inches(15, 10, forward=True)
    # real = plt.plot(todayPriceList, label='real')
    # pred = plt.plot(predictedPriceList, label='predicted')

    # if len(buys) > 0:
    #     plt.scatter(list(list(zip(*buys))[0]),
    #                 list(list(zip(*buys))[1]), c='#00ff00', s=30)
    # if len(sells) > 0:
    #     plt.scatter(list(list(zip(*sells))[0]),
    #                 list(list(zip(*sells))[1]), c='#ff0000', s=30)

    # caculate buy sell with difference predicted price
    buyWithDifference = []
    sellWithDifference = []
    buy_sell_difference = []

    rootPredictedPriceList = predictedPriceList[0]
    for index, predictedValue in enumerate(predictedPriceList):
        if index > 0:
            if predictedValue > predictedPriceList[index - 1]:
                buyWithDifference.append(
                    (index - 1, predictedPriceList[index - 1]))
                buy_sell_difference.append('buy')
            else:
                sellWithDifference.append(
                    (index - 1, predictedPriceList[index - 1]))
                buy_sell_difference.append('sell')

    # plt.scatter(list(list(zip(*buyWithDifference))[0]),
    #             list(list(zip(*buyWithDifference))[1]), c='#00ff00', s=10)
    # plt.scatter(list(list(zip(*sellWithDifference))[0]),
    #             list(list(zip(*sellWithDifference))[1]), c='#ff0000', s=10)

    # plt.legend(['Real', 'Predicted', 'Buy', 'Sell',
    #             'buyWithDifference', 'sellWithDifference'])

    # caculate accuracy
    # print('buyWithDifference', len(buyWithDifference))
    # print('sellWithDifference', len(sellWithDifference))
    # print('todayPriceList', len(todayPriceList))

    def caculate_accuracy_buy_sell_with_delay_session(deal_type, delay_session):
        total_caculate_deal = 0
        true_deal = 0
        false_deal = 0
        if(deal_type == 'buy'):
            list_data = buyWithDifference
        else:
            list_data = sellWithDifference
        # remove last item because list predict from 0 to length - 1
        length_real_data = len(todayPriceList) - 1
        for index, deal_predicted_price in list_data:
            # print(index, deal_predicted_price)
            index_has_delay_session = index + delay_session
            if index_has_delay_session <= length_real_data:
                total_caculate_deal += 1
                deal_real_price = todayPriceList[index]
                delay_session_price = todayPriceList[index_has_delay_session]
                # print('dealRealPrice', deal_real_price,
                #       'delay_sessionPrice', delay_session_price)
                if(deal_type == 'buy'):
                    if(delay_session_price > deal_real_price):
                        true_deal += 1
                    else:
                        false_deal += 1
                else:
                    if(delay_session_price < deal_real_price):
                        true_deal += 1
                    else:
                        false_deal += 1
        if total_caculate_deal > 0:
            print(deal_type, ': total deal: ', len(list_data), ', total caculate deal: ',
                  total_caculate_deal, ', true: ', true_deal, ', false: ', false_deal, ' accuracy: ',
                  round(true_deal/total_caculate_deal*100, 1), '%')
            return [deal_type, len(list_data), total_caculate_deal, true_deal, false_deal, round(true_deal/total_caculate_deal*100, 1)]
            # return [deal_type, len(list_data), total_caculate_deal, true_deal, false_deal,'no_signed_deal']
        else:
            print(deal_type, ': total deal: ', len(list_data), ', total caculate deal: ',
                  total_caculate_deal, ', true: ', true_deal, ', false: ', false_deal)
            return [deal_type, len(list_data), total_caculate_deal, true_deal, false_deal,'no_signed_deal']

    def caculate_accuracy_predicted_with_delay_session(delay_session):
        print('delay_session: ', delay_session)
        buy_result = caculate_accuracy_buy_sell_with_delay_session(
            'buy', delay_session)
        # sell_result = caculate_accuracy_buy_sell_with_delay_session('sell', delay_session)
        # print('total: total deal: ', buy_result[1]+sell_result[1], ', total caculate deal: ',
        #     buy_result[2] + sell_result[2], ', true: ', buy_result[3] +
        #     sell_result[3], ', false: ', buy_result[4] +
        #     sell_result[4], ' accuracy: ',
        #     ((buy_result[3] +
        #         sell_result[3])/(buy_result[2] + sell_result[2])*100), '%')

        return buy_result
    buy_result_list = {}
    for delay_session in predict_delay_session_list:
        buy_result = caculate_accuracy_predicted_with_delay_session(
            delay_session)
        buy_result_list[delay_session] = buy_result

    return {
        'options': options,
        'buy_result_list': buy_result_list,
        'predicted_price_list' : predictedPriceList,
        'buy_sell_difference' : buy_sell_difference
    }

    # plt.show()


def save_result(predict_result):
    info = get_info_from_options(predict_result['options'])
    base_name = info['base_name']
    moving_average = info['moving_average']
    epochs = info['epochs']
    predict_delay_session_list = info['predict_delay_session_list']
    result_name = info['result_name']
    if os.path.isfile(result_name) == False:
        result_file = open(result_name, "x")
        result_file.write("{}")
        result_file.close()
    pre_content = json.loads(open(result_name, "r").read())

    epochs_key = f'epochs_{epochs}'

    if (epochs_key in pre_content) == False:
        pre_content[epochs_key] = {}

    moving_average_key = f'moving_average_{moving_average}'
    if (moving_average_key in pre_content[epochs_key]) == False:
        pre_content[epochs_key][moving_average_key] = {}

    for delay_session in predict_delay_session_list:
        delay_session_key = f'delay_session_{delay_session}'
        pre_content[epochs_key][moving_average_key][delay_session_key] = predict_result['buy_result_list'][delay_session]

    pre_content['predicted_price_list'] = predict_result['predicted_price_list']
    pre_content['buy_sell_difference'] = predict_result['buy_sell_difference']
    result_file = open(result_name, "w+")
    result_file.write(json.dumps(pre_content, sort_keys=True))
    result_file.close()


def train_predict(input_config):
    # epochs_range = 2
    # moving_average_range = 5
    # predict_delay_session_list = [1, 3, 5, 10, 20]
    epochs_range = input_config['epochs_range']
    moving_average_range = input_config['moving_average_range']
    predict_delay_session_list = input_config['predict_delay_session_list']
    file_name = input_config['file_name']
    data_path = input_config['data_path']
    # for j in range(epochs_range):
    #     epochs = (j+1)*10
    epochs = epochs_range
    for i in range(moving_average_range):
        moving_average = (i+1)*10
        options = {
            'file_name': file_name,
            'epochs': epochs,
            'moving_average': moving_average,
            'predict_delay_session_list': predict_delay_session_list,
            'data_path': data_path
        }

        train_model(options)
        predict_result = predict_model(options)
        if predict_result != False:
            save_result(predict_result)


def find_max_value_matrix(value_matrix, options):
    info = get_info_from_options(options)
    predict_delay_session_list = info['predict_delay_session_list']
    moving_average_range = options.get('moving_average_range')

    return_value = {}
    # max moving_average
    for i in range(moving_average_range):
        moving_average = (i+1)*10
        moving_average_list = value_matrix[i]
        max_moving_average = max(moving_average_list)
        max_moving_average_key = f'max_moving_average_{moving_average}'
        return_value[max_moving_average_key] = max_moving_average
    # max delay_session
    for delay_session_index, delay_session in enumerate(predict_delay_session_list):
        delay_session_list = []
        for j in range(moving_average_range):
            delay_session_list.append(value_matrix[j][delay_session_index])
        max_delay_session = max(delay_session_list)
        max_delay_session_key = f'max_delay_session_{delay_session}'
        return_value[max_delay_session_key] = max_delay_session
    # max value_matrix
    max_value_matrix = {
        'value': value_matrix[0][0],
        'delay_session': predict_delay_session_list[0],
        'moving_average': 10
    }

    for delay_session_index, delay_session in enumerate(predict_delay_session_list):
        for k in range(moving_average_range):
            if value_matrix[k][delay_session_index] > max_value_matrix['value']:
                moving_average = (k+1)*10
                max_value_matrix = {
                    'value': value_matrix[k][delay_session_index],
                    'delay_session': delay_session,
                    'moving_average': moving_average
                }
    return_value['max_value_matrix'] = max_value_matrix
    return_value['value_matrix'] = value_matrix
    return_value['moving_average_range'] = moving_average_range
    return_value['predict_delay_session_list'] = predict_delay_session_list
    return return_value


def view_statistic_result(statistic_result, options):
    info = get_info_from_options(options)
    base_name = info['base_name']
    moving_average = info['moving_average']
    epochs = info['epochs']
    predict_delay_session_list = info['predict_delay_session_list']
    result_name = info['result_name']
    json_path = info['json_path']
    template_file = open(f'{os.getcwd()}/web/template/template.html', "r")
    content = template_file.read()
    template_file.close()

    result_file_name = f'{os.getcwd()}/web/html/{base_name}.html'
    result_file = open(result_file_name, "w")
    result_file.write(
        f'{content}<script>  var data = {statistic_result}; index.build(data);</script>')
    result_file.close()

    result_json_file_name = json_path
    result_json_file = open(result_json_file_name, "w")
    result_json_file.write(json.dumps(statistic_result, sort_keys=True))
    result_json_file.close()

    # print(statistic_result)

    # url = "file://"+result_file_name
    # webbrowser.open(url, new=1)


def statistic_result(input_config):
    # epochs_range = 2
    # moving_average_range = 5
    # predict_delay_session_list = [1, 3, 5, 10, 20]
    epochs_range = input_config['epochs_range']
    moving_average_range = input_config['moving_average_range']
    predict_delay_session_list = input_config['predict_delay_session_list']
    file_name = input_config['file_name']
    list_statistic = {}
    # for j in range(epochs_range):
    #     epochs = (j+1)*10
    epochs = epochs_range
    value_matrix = np.zeros(
        (moving_average_range, len(predict_delay_session_list)))
    for i in range(moving_average_range):
        moving_average = (i+1)*10
        options = {
            'file_name': file_name,
            'epochs': epochs,
            'moving_average': moving_average,
            'predict_delay_session_list': predict_delay_session_list,
            'moving_average_range': moving_average_range
        }
        info = get_info_from_options(options)
        moving_average = info['moving_average']
        epochs = info['epochs']
        predict_delay_session_list = info['predict_delay_session_list']
        result_name = info['result_name']
        pre_content = json.loads(open(result_name, "r").read())

        epochs_key = f'epochs_{epochs}'
        moving_average_key = f'moving_average_{moving_average}'
        for delay_session_index, delay_session in enumerate(predict_delay_session_list):
            delay_session_key = f'delay_session_{delay_session}'
            ratio_predicted = pre_content[epochs_key][moving_average_key][delay_session_key][-1]
            value_matrix[i][delay_session_index] = ratio_predicted if ratio_predicted!='no_signed_deal' else -1

    statistic_result = find_max_value_matrix(
        value_matrix.tolist(), options)
    list_statistic[epochs] = statistic_result

    statistic_result['buy_sell_difference'] = pre_content['buy_sell_difference']

    view_statistic_result(list_statistic, options)

    return list_statistic   


def main(options):
    input_config = options.get('input_config') or {
        'epochs_range': 10,
        'moving_average_range': 5,
        'predict_delay_session_list': [1, 3, 5, 10, 20],
        'file_name': 'AAA.csv',
    }
    train_predict(input_config)
    list_statistic = statistic_result(input_config)
    return list_statistic
