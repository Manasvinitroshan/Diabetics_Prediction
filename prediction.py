# prediction.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

def load_and_train(path='diabetes.csv'):
    # 1) Load data
    df = pd.read_csv(path)
    X = df.drop(columns='Outcome', axis=1)
    y = df['Outcome']

    # 2) Scale features
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, stratify=y, random_state=2
    )

    # 4) Fit SVM
    model = svm.SVC(kernel='linear')
    model.fit(X_train, y_train)

    # 5) Evaluate
    print("Training accuracy: ", accuracy_score(model.predict(X_train), y_train))
    print("Testing  accuracy: ", accuracy_score(model.predict(X_test),  y_test))

    return model, scaler

if __name__ == '__main__':
    clf, scaler = load_and_train('diabetes.csv')

    # 6) Serialize with pickle
    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ Model and scaler saved to disk as .pkl files.")
