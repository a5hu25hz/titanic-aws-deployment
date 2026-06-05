import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    print("Loading data...")
    df = pd.read_csv('titanic.csv')

    # Select the columns our API will use
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    target = 'Survived'
    data = df[features + [target]].copy()

    print("Preprocessing data...")
    # Fill missing values with medians to prevent training errors
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # Convert Categorical 'Sex' into a binary 'IsMale' feature
    data['IsMale'] = (data['Sex'] == 'male').astype(int)
    data = data.drop('Sex', axis=1)

    # Force strict feature ordering exactly as FastAPI expects it
    feature_order = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'IsMale']
    X = data[feature_order]
    y = data[target]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and fit the Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Verify accuracy
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy on Test Set: {acc * 100:.2f}%")

    # Save the artifacts into the app/ directory
    os.makedirs('app', exist_ok=True)
    joblib.dump(model, 'app/titanic_model.pkl')
    joblib.dump(scaler, 'app/model_scaler.pkl')
    print("Success! Artifacts saved to the app/ directory.")

if __name__ == "__main__":
    train_model()