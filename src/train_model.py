import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from preprocess import load_data

def train_rf(data_path, model_path):
    X, y = load_data(data_path)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )

    model.fit(X, y)
    joblib.dump(model, model_path)
    print("Random Forest model saved successfully.")

if __name__ == "__main__":
    train_rf("../data/lymphography.csv", "../models/random_forest.pkl")
