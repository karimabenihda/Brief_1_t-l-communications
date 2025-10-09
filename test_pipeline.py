import pandas as pd
from sklearn.model_selection import train_test_split


def test_split_dimension():
    data = pd.read_csv("data.csv")
    X=data.drop(columns=["Churn"])
    y=data["Churn"]

    assert len(X)==len(y),"incoherence: X={len(X)}, y={len(y)}"
    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.2,random_state=42
    )
    assert len(X_train)==len(y_train), f"Incohérence train : X_train={len(X_train)}, y_train={len(y_train)}"

# output: 
# ✅ Résultat : PASSED → ton test est réussi, tout est cohérent.
# Le [100%] indique que 100 % des tests ont passé (1 sur 1).    

