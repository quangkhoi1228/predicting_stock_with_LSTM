import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from util import csv_to_dataset, history_points

model = load_model('technical_model.h5')

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(
    'VN_INDEX_daily.csv')
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

start = 0
end = -1
# end = 20
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
    print(x, '/', length)


print(f"buys: {len(buys)}")
print(f"sells: {len(sells)}")


def compute_earnings(buys_, sells_):
    purchase_amt = 10
    stock = 0
    balance = 0
    while len(buys_) > 0 and len(sells_) > 0:
        if buys_[0][0] < sells_[0][0]:
            # time to buy $10 worth of stock
            balance -= purchase_amt
            stock += purchase_amt / buys_[0][1]
            buys_.pop(0)
        else:
            # time to sell all of our stock
            balance += stock * sells_[0][1]
            stock = 0
            sells_.pop(0)
    print(f"earnings: ${balance}")


# we create new lists so we dont modify the original
compute_earnings([b for b in buys], [s for s in sells])


plt.gcf().set_size_inches(15, 10, forward=True)
real = plt.plot(todayPriceList, label='real')
pred = plt.plot(predictedPriceList, label='predicted')

if len(buys) > 0:
    plt.scatter(list(list(zip(*buys))[0]),
                list(list(zip(*buys))[1]), c = '#00ff00', s = 30)
if len(sells) > 0:
    plt.scatter(list(list(zip(*sells))[0]),
                list(list(zip(*sells))[1]), c = '#ff0000', s = 30)


# caculate buy sell with difference predicted price
buyWithDifference=[]
sellWithDifference=[]

rootPredictedPriceList=predictedPriceList[0]
for index, predictedValue in enumerate(predictedPriceList):
    if index > 0:
        if predictedValue > predictedPriceList[index - 1]:
            buyWithDifference.append(
                (index - 1, predictedPriceList[index - 1]))
        else:
            sellWithDifference.append(
                (index - 1, predictedPriceList[index - 1]))

plt.scatter(list(list(zip(*buyWithDifference))[0]),
            list(list(zip(*buyWithDifference))[1]), c='#00ff00', s=10)
plt.scatter(list(list(zip(*sellWithDifference))[0]),
            list(list(zip(*sellWithDifference))[1]), c='#ff0000', s=10)

plt.legend(['Real', 'Predicted', 'Buy', 'Sell',
            'buyWithDifference', 'sellWithDifference'])


# caculate accuracy
print('buyWithDifference', len(buyWithDifference))
print('sellWithDifference', len(sellWithDifference))
print('todayPriceList', len(todayPriceList))


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

    print(deal_type, ': total deal: ', len(list_data), ', total caculate deal: ',
          total_caculate_deal, ', true: ', true_deal, ', false: ', false_deal, ' accuracy: ',
          (true_deal/total_caculate_deal*100), '%')
    return [deal_type, len(list_data), total_caculate_deal, true_deal, false_deal]

def caculate_accuracy_predicted_with_delay_session(delay_session):
    print('delay_session: ',delay_session)
    buy_result = caculate_accuracy_buy_sell_with_delay_session('buy', delay_session)
    # sell_result = caculate_accuracy_buy_sell_with_delay_session('sell', delay_session)
    # print('total: total deal: ', buy_result[1]+sell_result[1], ', total caculate deal: ',
    #     buy_result[2] + sell_result[2], ', true: ', buy_result[3] +
    #     sell_result[3], ', false: ', buy_result[4] +
    #     sell_result[4], ' accuracy: ',
    #     ((buy_result[3] +
    #         sell_result[3])/(buy_result[2] + sell_result[2])*100), '%')
caculate_accuracy_predicted_with_delay_session(1)
caculate_accuracy_predicted_with_delay_session(3)
caculate_accuracy_predicted_with_delay_session(5)
caculate_accuracy_predicted_with_delay_session(10)
plt.show()
