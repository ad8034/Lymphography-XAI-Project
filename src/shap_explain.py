import shap
import joblib
import pandas as pd
from preprocess import load_data

def shap_analysis(data_path, model_path):
    X, y = load_data(data_path)
    model = joblib.load(model_path)

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Global explanation (class-wise)
    shap.plots.bar(shap_values)
    shap.plots.beeswarm(shap_values[:, :, 0])  # Class 0 example

    # Local explanation
    shap.plots.waterfall(shap_values[0, :, 0])

if __name__ == "__main__":
    shap_analysis("../data/lymphography.csv", "../models/random_forest.pkl")
