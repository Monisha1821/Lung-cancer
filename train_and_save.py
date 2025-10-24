# train_and_save.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import json

# 1. load dataset (make sure file is in same folder)
df = pd.read_csv('survey lung cancer.csv')

# 2. fix obvious mappings you already used
df['GENDER'] = df['GENDER'].map({'M':0, 'F':1})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES':1, 'NO':0})

# 3. prepare X and y
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# 4. split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. fit LabelEncoders for object columns (store them)
encoders = {}
for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le
    else:
        # numeric columns keep as-is
        pass

# 6. train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. evaluate (optional quick check)
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8. save model + encoders + feature order
joblib.dump(model, 'model.joblib')
joblib.dump(encoders, 'encoders.joblib')

# save feature names in order (so the app knows expected order)
features = list(X.columns)
with open('features.json','w') as f:
    json.dump(features, f)

print("Saved model.joblib, encoders.joblib, features.json")
