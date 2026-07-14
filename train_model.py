import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

    # One-hot encode categoricals (same as original notebook)
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    X = df.drop(columns=["Exited"])
    y = df["Exited"].values
    return X, y


def build_model(input_dim: int) -> Sequential:
    model = Sequential([
        Dense(32, activation="relu", input_dim=input_dim),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main(csv_path: str, out_dir: str):
    X, y = load_and_prepare(csv_path)
    feature_columns = list(X.columns)  # save exact column order for inference

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    scaler = StandardScaler()
    X_train_trf = scaler.fit_transform(X_train)
    X_test_trf = scaler.transform(X_test)  # use the SAME scaler, transform only

    model = build_model(input_dim=X_train_trf.shape[1])
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    history = model.fit(
        X_train_trf, y_train,
        batch_size=32,
        epochs=200,
        verbose=1,
        validation_split=0.2,
        callbacks=[early_stop],
    )

    # Evaluate correctly: threshold the probability, don't argmax it
    y_pred_prob = model.predict(X_test_trf)
    y_pred = (y_pred_prob > 0.5).astype(int).ravel()

    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # --- Save everything the Streamlit app needs ---
    model.save(f"{out_dir}/churn_model.keras")

    with open(f"{out_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{out_dir}/feature_columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

    print(f"\nSaved model, scaler, and feature_columns to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Churn_Modelling.csv")
    parser.add_argument("--out", default=".", help="Output directory for artifacts")
    args = parser.parse_args()
    main(args.data, args.out)
