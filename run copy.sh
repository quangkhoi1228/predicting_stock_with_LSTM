pip3 install virtualenv
virtualenv envs
source envs/bin/activate
pip install -r requirements.txt --use-feature=2020-resolver
python save_data_to_csv.py MSFT daily #symbol = IBM, unit = daily
python tech_ind_model.py


source envs/bin/activate
python trading_algo.py

python tech_ind_model.py
python trading_algo_root_1.py

python base_data_excel_handle.py 


cd /Volumes/home/ProjectAIFinanceLearn/stock-trading-ml
source envs/bin/activate


python detect_all_symbol.py


python view_all_symbol.py 


cd /Volumes/home/ProjectAIFinanceLearn/predicting_stock_with_LSTM
source envs/bin/activate
python master_file.py 