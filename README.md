# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
Program to implement the linear regression using gradient descent.
Developed by: santhosh kumar B 
RegisterNumber:  212223230193
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
            return theta
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")


## Output:
![image](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/d8130240-00d9-4a13-958f-dd5d72f3a8cc)

![image](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/3dd917a4-11d6-4109-b277-4d1c144df82d)

![image](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/45014662-70a1-4913-9d34-0470cd52a2d9)

![image](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/018618ad-2127-43bb-bcd7-093e8852a65d)

![image](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/6ee40660-62fd-4d24-b64e-272226cfe378)













## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
