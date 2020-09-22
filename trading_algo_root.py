import numpy as np
from keras.models import load_model
from util import csv_to_dataset, history_points

model = load_model('technical_model.h5')

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('GSPC.csv')
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
# end = -1
end = 200
x = 0

todayPriceList = []
predictedPriceList = []

for ohlcv, ind in zip(ohlcv_test[start: end], tech_ind_test[start: end]):
    normalised_price_today = ohlcv[-1][0]
    normalised_price_today = np.array([[normalised_price_today]])
    price_today = y_normaliser.inverse_transform(normalised_price_today)
    ohlcv_predict = np.array(ohlcv,ndmin=3) #fix dimension
    ind_predict = np.array(ind,ndmin=2) #fix dimension
    predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([ohlcv_predict, ind_predict])))
    todayPriceList.append(price_today[0][0])
    predictedPriceList.append(float(predicted_price_tomorrow))
    delta = predicted_price_tomorrow - price_today
    if delta > thresh:
        buys.append((x, price_today[0][0]))
    elif delta < -thresh:
        sells.append((x, price_today[0][0]))
    x += 1
    print(x)
  

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

import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)
# print('unscaled_y_test[start:end]',unscaled_y_test[start:end].shape)
# real = plt.plot(unscaled_y_test[start:end], label='real')
# real = plt.plot(ohlcv_test[start:end], label='real')
real = plt.plot(todayPriceList, label='real')
# pred = plt.plot(y_test_predicted[start:end], label='predicted')
pred = plt.plot(predictedPriceList, label='predicted')
# print('real',unscaled_y_test[start:end])
# print('pred',y_test_predicted[start:end])
if len(buys) > 0:
    plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00', s=30)
    # plt.scatter(list(list(zip(*buys))[0]), unscaled_y_test[start:end][list(list(zip(*buys))[0])], c='#00ff00', s=10)
if len(sells) > 0:
    plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000', s=30)
    # plt.scatter(list(list(zip(*sells))[0]), unscaled_y_test[start:end][list(list(zip(*sells))[0])], c='#ff0000', s=10)
# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted', 'Buy', 'Sell'])

plt.show()