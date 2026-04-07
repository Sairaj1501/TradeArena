import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)

    df = df.dropna().reset_index(drop=True)

    # Price
    df["price"] = df["close"]

    # df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y %H:%M")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    df = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()

    # ✅ FIX 2: CREATE PRICE HERE (before RSI)
    df["price"] = df["close"]

    # RSI (14)
    window = 14
    delta = df["price"].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Moving Average (10)
    df["ma"] = df["price"].rolling(window=10).mean()

    # Trend
    def get_trend(row):
        if row["price"] > row["ma"]:
            return "bullish"  # Match the yaml
        elif row["price"] < row["ma"]:
            return "bearish"  # Match the yaml
        else:
            return "sideways"

    df["trend"] = df.apply(get_trend, axis=1)

    df = df.dropna().reset_index(drop=True)

    return df.to_dict(orient="records")