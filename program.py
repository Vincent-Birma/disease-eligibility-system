import pandas as pd 
import joblib as jb 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

disease = pd.read_csv('symptom.csv')

le = OrdinalEncoder()
disease[['Fever','Cough','Fatigue','Gender','Cholesterol Level','Blood Pressure']] =le.fit_transform(disease[['Fever','Cough','Fatigue','Gender','Cholesterol Level','Blood Pressure']])

X = disease[['Fever','Cough','Fatigue','Age','Gender','Cholesterol Level','Blood Pressure']]
y = disease['Outcome Variable']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state=40)

model = RandomForestClassifier(n_estimators= 100,random_state=40)
model.fit(X_train,y_train)

jb.dump(model,'disease.joblib')