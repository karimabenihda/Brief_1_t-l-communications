# pipeline.py : fonctions de préparation et modélisation
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,confusion_matrix,ConfusionMatrixDisplay
import xgboost as xgb
import pickle


# def load_data():
#     data = pd.read_csv("data_clean.csv")
#     print({data.shape})
#     return data
# load_data()

def Normalisation(): 
    data = pd.read_csv("data_clean.csv")
    encoder = LabelEncoder()

    cols = [
        'gender','Partner','Dependents','PhoneService','MultipleLines',
        'InternetService','OnlineSecurity','DeviceProtection','TechSupport',
        'StreamingTV','StreamingMovies','Contract','PaperlessBilling','OnlineBackup',
        'PaymentMethod','Churn'
    ]

    # Encode all categorical columns
    for col in cols:
        data[col] = encoder.fit_transform(data[col])


# def split_scale():
#     Normalisation()
    # data = pd.read_csv("data_clean.csv")
    # print(data.dtypes)

    X = data.drop(columns=['Churn','customerID'])
    y = data['Churn']
    # print(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler=MinMaxScaler()
    X_train_scaled=scaler.fit_transform(X_train.select_dtypes(include=['int64','float64']))
    # # print(X_train_scaled)
    X_test_scaled=scaler.fit_transform(X_test.select_dtypes(include=['int64','float64']))
    # # print(X_test_scaled)
    # selector=VarianceThreshold(threshold=0.01)
    # X_reduced=selector.fit_transform(X_train_scaled)
    # print(X_reduced)
   
    # print(" X_train_scaled shape:", X_train_scaled.shape)
    # print(" X_test_scaled shape:", X_test_scaled.shape)
    # print(" y_train shape:", y_train.shape)
    # print(" y_test shape:", y_test.shape)
    
    # model=RandomForestClassifier()

    # model=LogisticRegression( max_iter=2000,     
    # solver='liblinear',
    # random_state=42)

    # model.fit(X_train,y_train)

    # y_pred=model.predict(X_test)
    # print(y_pred,y_test)

    # return X_train_scaled, X_test_scaled, y_train, y_test, y_pred
    # print("accuracy: ",accuracy_score(y_test,y_pred))
        #output: accuracy:  0.815471965933286
    # print(confusion_matrix(y_test,y_pred))
        # output: [[933 103][157 216]]
    # print(classification_report(y_test,y_pred))
# split_scale()  


# XGBClassifier
    xgb_model=xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    eval_metric='logloss'
    )

    xgb_model.fit(X_train_scaled,y_train)
    with open("model.pkl","wb") as file:
        pickle.dump(xgb_model,file)

    with open("model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    y_pred = loaded_model.predict(X_test)

    y_pred=xgb_model.predict(X_test_scaled)
    print("accuracy: ",accuracy_score(y_test,y_pred))
        # output: accuracy:  0.815471965933286
    print(confusion_matrix(y_test,y_pred))
        # output: [[933 103][157 216]]
    print(classification_report(y_test,y_pred))

    scores=cross_val_score(xgb_model,X_train,y_train,cv=5,scoring='accuracy')
    print("scores:",scores)

    

    a=confusion_matrix(y_test,y_pred)
    print(a)
    disp=ConfusionMatrixDisplay(confusion_matrix=a,display_labels=["No","Yes"])
    disp.plot(cmap="Blues")
    plt.title('matrice de confusion ')
    # plt.show()

Normalisation()