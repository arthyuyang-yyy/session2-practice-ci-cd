import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


DATA_FILE = "training_data.csv"
MODEL_FILE = "linear_model.pkl"
METRICS_FILE = "linear_model.txt"


def main():
    data = pd.read_csv(DATA_FILE)

    X = data[["YearsExperience"]]
    y = data["Salary"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)

    coefficient = model.coef_[0]
    intercept = model.intercept_

    with open(METRICS_FILE, "w", encoding="utf-8") as output_file:
        output_file.write(f"Coefficient: {coefficient}\n")
        output_file.write(f"Intercept: {intercept}\n")


if __name__ == "__main__":
    main()
