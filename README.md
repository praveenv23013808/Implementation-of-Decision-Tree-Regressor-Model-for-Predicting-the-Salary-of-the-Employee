# EX 07-Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
## DATE: 18-03-2024
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sriram.v
RegisterNumber:  212222103002
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
Data.Head():

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/143114486/ccf52b5d-069a-4577-8a4f-53a186e2a0b0)

Data.info():

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/143114486/f13e5cd6-60e5-460a-aca5-25333b658b82)

isnull() and sum():

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/143114486/d24469dd-91dc-4fb9-887d-5e977379d855)

Data.Head() for salary:

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/143114486/5bbb7d7d-afdd-4518-aa44-4748430ff39a)

MSE Value:

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/143114486/b3bfef24-f3a7-4f44-bdfe-96af43c3b7ee)

r2 Value:

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/143114486/6cbd9c8f-b583-4c06-b633-36518138451b)

Data Prediction:

![image](https://github.com/Darkwebnew/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/143114486/5e32a429-c680-47d9-a196-b5cab4d3f3e9)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
