# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: santhosh kumar B
RegisterNumber: 212223230193 
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))
...


## Output:
![image 2024-02-22 212223 ml1](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/d8b99a3d-474f-42aa-a5a8-e1e74e20835b)
![image 2024-02-22 212340 ml 2](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/85d683d5-8ff1-4d25-8b3a-77a6dd3920ad)
![image 2024-02-22 212340](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/e0ccc771-54f1-45d7-a8dc-e4b4a780ca2d)
![image 2024-02-22 212601](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/2d42ce48-255d-4e99-9918-f7f08568eaa1)
![image 2024-02-22 212736](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/d2c7fecb-c460-4509-8d72-d6f45c40f4ed)
![image2024-02-22 213026](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/3fd2e2ba-f5ee-4690-ae3c-5b7f00b11b3d)
![image 2024-02-22 212918](https://github.com/Santhoshstudent/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145446853/fd5ebce2-b342-4314-ac30-059cf0fa8576)









## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
