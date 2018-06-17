# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 17:09:08 2018

@author: 1000091
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
boston = load_boston()



#prepare the DF for independant variable
df_x = pd.DataFrame(boston.data,columns=boston.feature_names)
print (df_x.head(15))
print (df_x.shape)
df_y= pd.DataFrame(boston.target)
print(df_y.head(15))

names= [i for i in list(df_x)]
print (names)


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.30, random_state = 5)
print (x_train.head())

regression = LinearRegression()
# Fit the linear regression model to the training data set.
regression.fit(x_train,y_train)
y_pred= regression.predict(x_test)
LinearRegression(copy_X= True, fit_intercept= True, n_jobs= 1, normalize= False)

print(regression.intercept_)

print ("Coefficients \n", regression.coef_)
print ("MSE %.2f" % np.mean((regression.predict(x_test)-y_test)**2))
print ("Variance Score %.2f" % regression.score(x_test, y_test))

Coef_List = regression.coef_[0].tolist()
print (Coef_List)

df= pd.DataFrame({"Coefficients":Coef_List, "Feature Name":names})

print (df)
#print(pd.DataFrame(zip(names,regression.coef_[0].tolist()),columns=["names", "coefficient"]))
print ("plot the predicted x_test and y_test values:=====\n\n")

plt.scatter(regression.predict(x_test),y_test)
plt.show()

plt.scatter(y_test, y_pred)
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel("True price")
plt.ylabel("Predicted price")
plt.title("True Prices vs Predicted prices")
plt.show()

plt.figure(figsize=(9,6))
plt.scatter(regression.predict(x_train), regression.predict(x_train) - y_train, c='b', s=40, alpha=0.5)
plt.scatter(regression.predict(x_test), regression.predict(x_test) - y_test, c='g', s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=50)
plt.ylabel('Residuals')
plt.title('Residual plot including training(blue) and test(green) data')
plt.show()

print ("Y_test:\n\n",y_test)
print ("Y_Predicted:\n\n",y_pred)

