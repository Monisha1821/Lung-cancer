# app.py
from flask import Flask, render_template, request
import joblib
import json
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model, encoders, feature order
model = joblib.load('model.joblib')
encoders = joblib.load('encoders.joblib')  # may be empty dict if none
with open('features.json','r') as f:
    features = json.load(f)

# Helper to preprocess single input row
def preprocess_input(form):
    # Build a list in the same order as 'features'
    row = []
    for feat in features:
        val = form.get(feat)
        # If user left empty, try default 0
        if val is None or val == '':
            val = 0
        # We need to match training dtype/encoder:
        if feat in encoders:
            le = encoders[feat]
            # transform expects same categories seen before. If unknown, try to add safe fallback:
            try:
                encoded = int(le.transform([str(val)])[0])
            except Exception:
                # fallback: if unseen label, try to map to 0
                encoded = 0
            row.append(encoded)
        else:
            # numeric column -> convert to float/int
            try:
                row.append(float(val))
            except:
                row.append(0.0)
    return np.array(row).reshape(1, -1)

@app.route('/', methods=['GET'])
def index():
    # Provide form with feature names. The index.html will create inputs for each feature.
    # Also pass choices for certain fields (you can customize).
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    X = preprocess_input(request.form)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max() if hasattr(model, "predict_proba") else None
    result_text = "YES (Has Lung Cancer)" if pred==1 else "NO (Does NOT have Lung Cancer)"
    return render_template('result.html', prediction=result_text, probability=prob)

if __name__ == '__main__':
    app.run(debug=True)
