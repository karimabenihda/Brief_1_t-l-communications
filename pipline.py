# pipeline.py : fonctions de préparation et modélisation
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.feature_selection import VarianceThreshold


def load_data():
    data = pd.read_csv("data_clean.csv")
    print({data.shape})
    return data
load_data()

def Normalisation():
    data = pd.read_csv("data_clean.csv")
    encoder = LabelEncoder()

    cols = [
        'gender','Partner','Dependents','PhoneService','MultipleLines',
        'InternetService','OnlineSecurity','DeviceProtection','TechSupport',
        'StreamingTV','StreamingMovies','Contract','PaperlessBilling',
        'PaymentMethod','Churn'
    ]

    # Encode all categorical columns
    for col in cols:
        data[col + '_num'] = encoder.fit_transform(data[col])


# Normalisation()

def split_scale():
    data = pd.read_csv("data_clean.csv")

    # Split data (after encoding)
    X = data.drop(columns=['Churn'])
    y = data['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric columns
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=['int64', 'float64']))
    X_test_scaled = scaler.transform(X_test.select_dtypes(include=['int64', 'float64']))

    selector = VarianceThreshold(threshold=0.01)
    X_reduced = selector.fit_transform(X_train_scaled)
    print(X_reduced)
    print("🔹 X_train_scaled shape:", X_train_scaled.shape)
    print("🔹 X_test_scaled shape:", X_test_scaled.shape)
    print("🔹 y_train shape:", y_train.shape)
    print("🔹 y_test shape:", y_test.shape)

    return X_train_scaled, X_test_scaled, y_train, y_test

split_scale()  



