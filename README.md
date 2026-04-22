# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize parameters m,bm, bm,b and normalize input data.
2.Predict output using ypred=mX+by_{pred} = mX + bypred​=mX+b.
3.Compute gradients of loss w.r.t. mmm and bbb.
4.Update m,bm, bm,b using gradient descent until convergence.

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: Krithik kiran S
RegisterNumber: 212225230145
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Startup.csv")

# Select one feature (R&D Spend) and target (Profit)
X = data['R&D Spend'].values
y = data['Profit'].values

# Normalize (important for gradient descent)
X = (X - X.mean()) / X.std()

# Initialize parameters
m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

# Gradient Descent
for i in range(epochs):
    y_pred = m * X + b
    
    # Gradients
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # Update
    m = m - learning_rate * dm
    b = b - learning_rate * db

print("Slope (m):", m)
print("Intercept (b):", b)

# Predictions for plotting
y_pred = m * X + b

# Plot
plt.scatter(X, y)
plt.plot(X, y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")

plt.show() 

```

## Output:

<img width="1042" height="634" alt="image" src="https://github.com/user-attachments/assets/84e9d0d8-e24d-4f2c-9992-db810cde7a2c" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
