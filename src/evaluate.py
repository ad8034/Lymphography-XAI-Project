import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data

def geometric_mean(recall):
    return np.prod(recall) ** (1 / len(recall))

def evaluate_model(data_path):
    X, y = load_data(data_path)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    model = RandomForestClassifier(random_state=42)

    acc, prec, rec, f1, gm = [], [], [], [], []

    for train, test in skf.split(X, y):
        model.fit(X.iloc[train], y.iloc[train])
        y_pred = model.predict(X.iloc[test])

        acc.append(accuracy_score(y.iloc[test], y_pred))
        prec.append(precision_score(y.iloc[test], y_pred, average="macro"))
        rec.append(recall_score(y.iloc[test], y_pred, average="macro"))
        f1.append(f1_score(y.iloc[test], y_pred, average="macro"))

    print("Accuracy:", np.mean(acc))
    print("Precision:", np.mean(prec))
    print("Recall:", np.mean(rec))
    print("F1:", np.mean(f1))

if __name__ == "__main__":
    evaluate_model("../data/lymphography.csv")
