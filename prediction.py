# Diabetics_Prediction.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

def load_and_train(path='diabetes.csv'):
    # Load the dataset
    diabetes_dataset = pd.read_csv(path)

    # Separate features and target
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    y = diabetes_dataset['Outcome']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=2
    )

    # Train SVM classifier
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(classifier.predict(X_train), y_train)
    test_acc  = accuracy_score(classifier.predict(X_test),  y_test)
    print(f'Accuracy score of the training data: {train_acc:.4f}')
    print(f'Accuracy score of the testing data:  {test_acc:.4f}')

    return classifier, scaler

if __name__ == '__main__':
    clf, scaler = load_and_train('diabetes.csv')
    # Save model and scaler
    joblib.dump(clf,    'diabetes_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model and scaler have been saved to disk.")
