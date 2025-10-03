# SGD-REGRESSOR-FOR-MULTIVARIATE-LINEAR-REGRESSION

## AIM :

To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## EQUIPMENTS REQUIRED :
1. Hardware – PCs

2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM :
1. Load the California housing dataset.Create a DataFrame, select features and multi-targets (AveOccup, target).Split into training and testing sets.

2. Standardize both input (X) and output (Y) data using StandardScaler.

3. Use SGDRegressor wrapped in MultiOutputRegressor to fit the training data.

4. Predict on test data, inverse transform predictions, and calculate Mean Squared Error (MSE).

## PROGRAM :
```
/*
SGD-REGRESSOR-FOR-MULTIVARIATE-LINEAR-REGRESSION

DEVELOPED BY : ANTONY ABISHEK IK
 
REGISTER NUMBER :  212223240009
*/
```

```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScalerdata=fetch_california_housing()
print(data)
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
df.tail()
df.info()
x=df.drop(columns=['AveOccup','target'])
y=df[['AveOccup','target']]
x.shape
y.shape
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=11)
x_train.shape
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
x_test=scaler_x.fit_transform(x_test)
y_train=scaler_y.fit_transform(y_train)
y_test=scaler_y.fit_transform(y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
Y_pred=multi_output_sgd.predict(x_test)
Y_pred
y_pred=scaler_y.inverse_transform(Y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error :",mse)
y_pred
```
## OUTPUT :

<img width="742" height="290" alt="image" src="https://github.com/user-attachments/assets/3df1857c-4987-407a-9432-341b6f9779b0" />

<img width="824" height="214" alt="image" src="https://github.com/user-attachments/assets/b90c5d79-9bc6-46b7-ae55-cb3a1f99c8ce" />

<img width="426" height="347" alt="image" src="https://github.com/user-attachments/assets/956e6c69-b659-492a-999c-d50cf5ab5ef7" />

<img width="107" height="32" alt="image" src="https://github.com/user-attachments/assets/fb5dfd8a-fb46-4e7c-9032-573fb3b4d778" />

<img width="106" height="29" alt="image" src="https://github.com/user-attachments/assets/6056e2d0-d368-4d62-a25d-65645bab8cf7" />

<img width="383" height="174" alt="image" src="https://github.com/user-attachments/assets/03551dd4-65a2-426f-a344-9d2c2def81ff" />

<img width="374" height="29" alt="image" src="https://github.com/user-attachments/assets/54a952fc-e654-4004-99e9-1f8507028d48" />

<img width="337" height="151" alt="image" src="https://github.com/user-attachments/assets/f57293ee-f518-48b0-87fa-d8b0f94d1cd2" />

## RESULT :

Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
