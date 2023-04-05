from flask import Flask

app = Flask(__name__)
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import os
os.chdir('/home/BVKiran/pro')
data=pd.read_csv('diabetes.csv')
data.columns
data.drop(columns=['Pregnancies','Age'],inplace=True)
x=data.drop('Outcome',axis=1)
y=data['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
k=SVC()
k.fit(x_train,y_train)
predictions=k.predict(x_test)
test=k.predict([[183,64,0,0,23,0.672]])
from flask import Flask,request,jsonify
app= Flask(__name__)
@app.route('/hello')
def new():
    return "Tarun the SnapStar"
@app.route('/<int:Glucose>/<int:BloodPressure>/<int:SkinThickness>/<int:Insulin>/<int:BMI>/<float:DiabetesPedigreeFunction>')
def test(Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction):
    p=[]
    p +=[Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction]

    arr=np.array([p])
    predict=k.predict(arr)
    arr=np.array([p])
    predict=k.predict(arr)

    if predict == [1]:
        result = {'result':'Yes'}
    else:
        result = {'result':"N0"}


    return result

