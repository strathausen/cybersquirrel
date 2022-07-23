from pandas import DataFrame
from typing import List
import numpy as np

# Used for the RSI thanks to
# https://stackoverflow.com/questions/57006437/calculate-rsi-indicator-from-pandas-dataframe
n = 14


def rma(x, n, y0):
    a = (n-1) / n
    ak = a**np.arange(len(x)-1, -1, -1)
    return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x) + 1)]


def create_features(data: DataFrame, symbol: str, window: int, lags: int):
    data["returns"] = np.log(data[symbol] / data[symbol].shift())
    df = data.copy()
    df["dir"] = np.where(df["returns"] > 0, 1, 0)
    df["sma"] = df[symbol].rolling(
        window).mean() - df[symbol].rolling(150).mean()
    df["boll"] = (df[symbol] - df[symbol].rolling(window).mean()
                  ) / df[symbol].rolling(window).std()
    df["min"] = df[symbol].rolling(window).min() / df[symbol] - 1
    df["max"] = df[symbol].rolling(window).max() / df[symbol] - 1
    df["mom"] = df["returns"].rolling(3).mean()
    df["vol"] = df["returns"].rolling(window).std()

    # RSI calculation
    df['change'] = df['close'].diff()
    df['gain'] = df.change.mask(df.change < 0, 0.0)
    df['loss'] = -df.change.mask(df.change > 0, -0.0)
    df['avg_gain'] = rma(df.gain[n+1:].to_numpy(), n,
                         np.nansum(df.gain.to_numpy()[:n+1])/n)
    df['avg_loss'] = rma(df.loss[n+1:].to_numpy(), n,
                         np.nansum(df.loss.to_numpy()[:n+1])/n)
    df['rs'] = df.avg_gain / df.avg_loss
    df['rsi_14'] = 100 - (100 / (1 + df.rs))

    df.dropna(inplace=True)
    # Adding feature lags
    features = ["dir", "sma", "boll", "min", "max", "mom", "vol", "rsi_14"]
    cols: List[str] = []
    for f in features:
        for lag in range(1, lags + 1):
            col = f"{f}_lag_{lag}"
            df[col] = df[f].shift(lag)
            cols.append(col)
    return cols, df
