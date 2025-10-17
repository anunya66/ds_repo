
"""
demo_predict.py
A safe script that tries to load /mnt/data/new_catboost_model.pkl and run a small prediction
If the CatBoost model requires the catboost package, the script will prompt how to proceed.
"""
import os, pickle
import numpy as np
import pandas as pd

MODEL_PATH = "/mnt/data/new_catboost_model.pkl"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Place it there to run predictions.")
        return
    try:
        with open(MODEL_PATH,"rb") as f:
            model = pickle.load(f)
        print("Model loaded:", type(model))
    except Exception as e:
        print("Failed to load model via pickle:", e)
        print("If the model was saved using CatBoost native format, install `catboost` and try loading.")
        return
    # Create dummy input depending on model interface
    try:
        # If model has a predict method that accepts 2D array
        X = np.zeros((2, len(getattr(model, "feature_names_", [])) or 4))
        preds = model.predict(X)
        print("Sample predictions:", preds)
    except Exception as e:
        print("Model prediction failed (this is a demo). Error:", e)

if __name__ == "__main__":
    main()
