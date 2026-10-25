"""
TASK 1: Linear Regression - House Price Prediction
----------------------------------------------------
Predicts house prices based on square footage, number of bedrooms,
and number of bathrooms.

NOTE: This script generates a realistic synthetic housing dataset so it
runs end-to-end with no external downloads. To use a real dataset instead
(e.g. a Kaggle housing CSV), replace the `generate_data()` call with:
    df = pd.read_csv("your_housing_data.csv")
as long as it has columns: sqft, bedrooms, bathrooms, price.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_data(n_samples=500, seed=42):
    """Create a synthetic but realistic housing dataset."""
    rng = np.random.default_rng(seed)

    sqft = rng.normal(1800, 650, n_samples).clip(400, 6000)
    bedrooms = rng.integers(1, 6, n_samples)
    bathrooms = rng.integers(1, 4, n_samples) + rng.choice([0, 0.5], n_samples)

    # Price driven by sqft, bedrooms, bathrooms + noise
    price = (
        50_000
        + sqft * 150
        + bedrooms * 12_000
        + bathrooms * 9_000
        + rng.normal(0, 25_000, n_samples)
    ).clip(50_000, None)

    return pd.DataFrame({
        "sqft": sqft.round(0),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "price": price.round(0),
    })


def main():
    df = generate_data()
    print("Sample of the dataset:")
    print(df.head(), "\n")

    X = df[["sqft", "bedrooms", "bathrooms"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Model coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature:10s}: {coef:,.2f}")
    print(f"  intercept : {model.intercept_:,.2f}\n")

    print("Evaluation on held-out test set:")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  R^2  : {r2:.4f}\n")

    # Example prediction
    example = pd.DataFrame({"sqft": [2200], "bedrooms": [3], "bathrooms": [2]})
    pred_price = model.predict(example)[0]
    print(f"Example prediction for a 2200 sqft, 3bd/2ba house: ${pred_price:,.0f}")

    # Plot: actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title("Task 1: Linear Regression - Actual vs Predicted House Prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig("task1_actual_vs_predicted.png", dpi=150)
    print("\nSaved plot to task1_actual_vs_predicted.png")


if __name__ == "__main__":
    main()
