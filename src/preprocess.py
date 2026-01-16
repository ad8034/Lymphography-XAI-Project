import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("class", axis=1)
    y = df["class"]
    return X, y
