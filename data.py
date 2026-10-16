"""Real-world streaming benchmark loaders. All streams keep their original temporal order."""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("STREAM_DATA_DIR")
if not DATA_DIR:
    _local_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    _parent_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    DATA_DIR = _local_data if os.path.exists(_local_data) else _parent_data


def load_stream(name, max_n=None):
    """Return X (float32, n x p), y (int 0/1), feature_names, cont (indices of continuous features)."""
    if name == "electricity":           # NSW electricity market (Elec2), 45,312 x 8
        df = pd.read_csv(os.path.join(DATA_DIR, "electricity.csv"))
        y = (df["class"].astype(str).str.upper() == "UP").astype(int).values
        # 'date' is a monotone time index, not a market variable: it is removed so that
        # no model can memorise position in the stream.
        X = df.drop(columns=["class", "date"])
    elif name == "weather":             # NOAA Bellevue weather, 18,159 x 8 (rain / no rain)
        df = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
        y = df["target"].astype(int).values
        X = df.drop(columns=["target"])
        X.columns = ["temp", "dew_pt", "sea_pres", "visib", "wind_avg", "wind_max", "temp_max", "temp_min"]
    elif name == "airlines":            # US flight delay stream, 539,383 x 7
        df = pd.read_csv(os.path.join(DATA_DIR, "airlines.csv"))
        y = df["Delay"].astype(int).values
        X = df.drop(columns=["Delay"])
    elif name == "covertype":           # Forest cover type, 581,012 x 54; Lodgepole pine (type 2) vs rest
        df = pd.read_csv(os.path.join(DATA_DIR, "covtype.csv"))
        tcol = df.columns[-1]
        y = (df[tcol].astype(int).values == 2).astype(int)
        X = df.drop(columns=[tcol])
    else:
        raise ValueError(name)
    names = list(X.columns)
    cont = {"electricity": [1, 2, 3, 4, 5, 6],          # all but day-of-week
            "weather": list(range(8)),
            "airlines": [5, 6],                         # Time, Length (others are nominal codes)
            "covertype": list(range(10))}[name]        # 10 cartographic variables (rest are one-hot)
    X = X.values.astype(np.float32)
    if max_n is not None:
        X, y = X[:max_n], y[:max_n]
    return X, y, names, cont
