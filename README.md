# codingtest

First run the generate_data.py to split the raw data into a separate csv for each stock and save them in the raw_stock folder. In the meantime, calculate some factor values for each stock to make training data and save them in the train_stock folder. (need delete the aaa file)

Then run the main.py to perform MLP model and get the predict value. Here I make the data from 2013/1/4 to 2019/12/30 as the train set and the data from 2020/1/6-2021/3/19 as the backtest set. The models are saved in the Modelsave folder and the predict data is saved in the predict folder. (need delete the aaa file)

Afterwards, run the predict_data.py to combine the predicted values of all stocks into the same csv.

Finally run the strategy.py to get the results of the backtest.
