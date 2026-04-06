import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
df = pd.read_csv("churn_data.csv")
df = df[[
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService",
    "OnlineSecurity",
    "PaymentMethod",
    "Partner",
    "Dependents",
    "Churn"
]]
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"],errors="coerce")
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].median())
df["Partner"] = df["Partner"].map({"Yes":1,"No":0})
df["Dependents"] =  df["Dependents"].map({"Yes":1,"No":0})
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})
df["OnlineSecurity"] = df["OnlineSecurity"].map({"Yes":1,"No":0,"No internet service":0})
df = pd.get_dummies(df,columns=["Contract","InternetService","PaymentMethod"],drop_first=True)
df.fillna(0,inplace=True)
int_cols = ["tenure","Dependents","Partner","OnlineSecurity"]
df[int_cols] = df[int_cols].astype(int)
x = df.drop("Churn",axis=1)
y = df["Churn"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)
model = LogisticRegression(class_weight='balanced',max_iter=1000)
model.fit(x_train,y_train)
y_prob = model.predict_proba(x_test)[:,1]
y_pred = (y_prob>0.7).astype(int)
score = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test,y_pred)
print("Accuracy:", score)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", cr)
pickle.dump(x.columns, open("feature_names.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scalar, open("scaler.pkl", "wb"))