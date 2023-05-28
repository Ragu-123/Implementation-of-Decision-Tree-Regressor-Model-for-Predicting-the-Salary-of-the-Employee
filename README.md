# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries.

2.Upload the dataset in the compiler and read the dataset.

3.Find head,info and null elements in the dataset.

4.Using LabelEncoder and DecisionTreeRegressor , find MSE and R2 of the dataset.

5.Predict the values and end the program.
```

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:RAGUNATH R 
RegisterNumber:212222240081  
*/
import pandas as pd
data=pd.read_csv("/content/Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## 1.data.head()
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113915622/4edb81cc-9457-4576-b227-9f9df4741f04)
## 2.data.info()
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113915622/7c920663-e2fa-4f49-bf1e-6edfa7162704)
## 3.isnull() and sum()
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113915622/dbb4d611-32f1-4ae2-b85e-fea7582d359f)
## 4.data.head() for salary
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113915622/103125f6-cdbf-486c-b85f-a4cb182873ea)
## 5.MSE value
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113915622/c69090be-3966-4eec-9076-09e359bab0f6)
## 6.r2 value
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113915622/39e35de8-17b5-49e0-b65b-76bdea0986f0)
## 7.data prediction
![image](https://github.com/Ragu-123/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/113915622/ad8a62d9-3640-4957-b79a-aa42751fb237)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
