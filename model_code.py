import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('Training.csv')
test = pd.read_csv('Testing.csv')

X_train = data.drop(columns=['Outcome'])
y_train = data['Outcome']
X_train.shape, y_train.shape

X_test = test.drop(columns=['Outcome'])
y_test = test['Outcome']
X_test.shape, y_test.shape

# Train a Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # You can modify these parameters
rf.fit(X_train, y_train)

# Predict on the test set
y_preds = rf.predict(X_test)
print('Accuracy score is', accuracy_score(y_test, y_preds))

# Save the trained Random Forest model
joblib.dump(rf, 'RandomForest.joblib')
