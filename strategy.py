from DNNModel import *
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List

from create_features import create_features
os.environ['F_ENABLE_ONEDNN_OPTS'] = '0'
plt.style.use("seaborn")
pd.set_option('display.float_format', lambda x: '%.5f' % x)


class CyberSquirrelStrategy:
    # Trading history and indicators
    raw_data: pd.DataFrame
    # Data with features and lags
    data: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame

    # How much of the data will be used for training
    split_ratio = 0.66

    # Feaures that are being used for training and predicting
    cols: List[str] = []

    # Cost for transactions
    ptc = 0.000059

    def __init__(self, symbol: str, window=50, lags=5):
        self.symbol = symbol
        self.window = window
        self.lags = lags

    def download_data(self):
        # TODO download data from yfinance?
        pass

    def read_data(self):
        pass

    def add_indicators(self):
        self.cols, self.data = create_features(self.raw_data, self.symbol,
                                               self.window, self.lags)

    def train_and_save(self):
        """Trains on the data and saves the model + parameters"""
        df = self.data
        cols = self.cols
        split = int(len(df) * self.split_ratio)
        train = df.iloc[:split].copy()
        self.train = train
        test = df.iloc[split:].copy()
        self.test = test

        # Feature scaling (Standardization)
        mu, std = train.mean(), train.std()
        train_s = (train - mu) / std

        # fitting a DNN model with 3 Hidden Layers (50 nodes each) and dropout
        # regularization
        set_seeds(100)
        model = create_model(hl=3, hu=50, dropout=True,
                             input_dim=len(cols))
        model.fit(x=train_s[cols], y=train["dir"], epochs=50,
                  verbose=False, validation_split=0.2, shuffle=False,
                  class_weight=cw(train))
        model.evaluate(train_s[cols], train["dir"])
        # pred = model.predict(train_s[cols])
        # print(pred)

        # standardization of test set features (with train set parameters!!!)
        test_s = (test - mu) / std
        model.evaluate(test_s[cols], test["dir"])
        # pred = model.predict(test_s[cols])
        test["proba"] = model.predict(test_s[cols])
        # 1. short where proba < 0.47
        test["position"] = np.where(test.proba < 0.47, -1, np.nan)
        # 2. long where proba > 0.53
        test["position"] = np.where(test.proba > 0.53, 1, test.position)
        # Only test during NY stock marked open hours
        test.index = test.index.tz_localize("UTC")
        test["NYTime"] = test.index.tz_convert("America/New_York")
        test["hour"] = test.NYTime.dt.hour
        # test.position.value_counts(dropna = False)
        test["strategy"] = test["position"] * test["returns"]
        test["creturns"] = test["returns"].cumsum().apply(np.exp)
        test["cstrategy"] = test["strategy"].cumsum().apply(np.exp)
        test["strategy_net"] = test.strategy - test.trades * self.ptc
        test["cstrategy_net"] = test["strategy_net"].cumsum().apply(np.exp)
        model.save(f"DNN_model_{self.symbol}")
        params = {"mu": mu, "std": std,
                  "window": self.window, "lags": self.lags}
        pickle.dump(params, open(f"params_{self.symbol}.pkl", "wb"))
        # test[["creturns", "cstrategy", "cstrategy_net"]].plot(figsize = (12, 8))
        # test[["creturns", "cstrategy"]].plot(figsize = (12, 8))
        # plt.show()
        # print(pred)
