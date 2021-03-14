# DATA-310---Lab-4

## Question 1: Regularization is defined as

B. The minimization of the sum of squared residuals subject to a constraint on the weights (aka coefficients). - Regularization is # necessary when the input features X are rank deficient or when there is strong multiple linear correlations among the input features. # Regularization tries to minimize the sum of the square residuals like a normal regression, but there is a constraint of the weights.

## Question 2: The regularization with the square of an L2 distance may improve the results compared to OLS when the number of features is higher than the number of observations.

True. - When the number of features is higher than the number of observations regularization is necessary because there is # rank deficiency.

## Question 3: The L1 norm always yields shorter distances compared to the Euclidean norm.

False. - L2 (or Ridge Regression) penalizes a model for having more or larger parameters, while L1 (or Lasso Regression) can# set some of the model coefficients to zero which basically removes those variables from the regression. So, depending on your data# and what it looks like L1 or L2 could yield a shorter distance.

## Question 4: Typically, the regularization is achieved by

D. minimizing the average of the squared residuals plus a penalty function whose input is the vector of coefficients. -

<img width="782" alt="Screen Shot 2021-03-14 at 11 36 20 AM" src="https://user-images.githubusercontent.com/74326062/111074715-a875ad80-84ba-11eb-9767-b6846a3629e0.png">

As you can see above all types of regression minimize the average squared residuals plus a function whose input is # the vector of coefficients.

## Question 5: A regularization method that facilitates variable selection (estimating some coefficients as zero) is

D. Lasso - Because Lasso has a penalty function that uses the absolute value of the vector of coefficients it can set certain# variable weights to 0, basically removing them from the regression.

## Question 6: Write your own Python code to import the Boston housing data set (from the sklearn library) and scale the data (not the target) by z-scores. If we use all the features with the Linear Regression to predict the target variable then the root mean squared error (RMSE) is: (your answer should include only the first 4 decimals that you get from the code)

from sklearn.datasets import load_boston

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error as MSE

import pandas as pdimport numpy as np

data = load_boston()x = pd.DataFrame(data=data.data, columns=data.feature_names)y = data.target

def zto(x): return (x-np.min(x))/(np.max(x)-np.min(x))xscaled = zto(x)xscaled

lm = LinearRegression()lm.fit(xscaled,y)yhat_lm = lm.predict(xscaled)

#RMSE

np.sqrt(np.mean((y-yhat_lm)^2)) = 4.679191295697282

## Question 7:On the Boston housing data set if we consider the Lasso model with 'alpha=0.03' then the 10-fold cross-validated prediction error is: (for the 10-fold cross-validation shuffle you should use random_state=1234, your final answer should include only the first 4 decimals that you get from the code)

from sklearn.linear_model import Lasso

from sklearn.model_selection import KFold

kf = KFold(n_splits=10, random_state=1234,shuffle=True)

df = pd.DataFrame(data=data.data, columns=data.feature_names)i = 0PE = []PE_train = []

for train_index, test_index in kf.split(df):

X_train = df.values[train_index

y_train = y[train_index]

X_test = df.values[test_index]

y_test = y[test_index]

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred_train = model.predict(X_train)

PE_train.append(MSE(y_train,y_pred_train))

PE.append(MSE(y_test, y_pred))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.sqrt(np.mean(PE_train)))) = The k-fold crossvalidated error rate on the train sets is: 4.70117379068021

print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.sqrt(np.mean(PE)))) = The k-fold crossvalidated error rate on the test sets is: 4.974181421491436

## Question 8: On the Boston housing data set if we consider the Elastic Net model with 'alpha=0.05' and 'l1_ratio=0.9' then the 10-fold cross-validated prediction error is: (for the 10-fold cross-validation shuffle you should use random_state=1234, your final answer should include only the first 4 decimals that you get from the code)

from sklearn.linear_model import ElasticNet

kf = KFold(n_splits=10, random_state=1234,shuffle=True)

df = pd.DataFrame(data=data.data, columns=data.feature_names)y = data.target

model = ElasticNet(alpha=0.01,l1_ratio=0.9)

i = 0

PE = []

PE_train = []

for train_index, test_index in kf.split(df):

X_train = df.values[train_index]

y_train = y[train_index]

X_test = df.values[test_index]

y_test = y[test_index]

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_pred_train = model.predict(X_train)

PE_train.append(MSE(y_train,y_pred_train))

PE.append(MSE(y_test, y_pred))

print('The k-fold crossvalidated error rate on the train sets is: ' + str(np.sqrt(np.mean(PE_train)))) = The k-fold crossvalidated error rate on the train sets is: 4.681045462862472

print('The k-fold crossvalidated error rate on the test sets is: ' + str(np.sqrt(np.mean(PE)))) = The k-fold crossvalidated error rate on the test sets is: 4.949410860688574

## Question 9: If we create all quadratic polynomial (degree=2) features based on the z-scores of the original features and then apply OLS, the root mean squared error is:

from sklearn.preprocessing import PolynomialFeatures

polynomial_features= PolynomialFeatures(degree=2)

x_poly = polynomial_features.fit_transform(xscaled)

lm = LinearRegression()

lm.fit(x_poly,y)

yhat_lm = lm.predict(x_poly)

#RMSE

np.sqrt(np.mean((y-yhat_lm)^2)) = 2.4482875016619703

## Question 10: If we create all quadratic polynomial (degree=2) features based on the z-scores of the original features and then apply the Ridge regression with alpha=0.1 and we create a Quantile-Quantile plot for the residuals then the result shows that the obtained residuals pretty much follow a normal distribution.

from sklearn.linear_model import Ridge

polynomial_features= PolynomialFeatures(degree=2)

x_poly = polynomial_features.fit_transform(xscaled)

model = Ridge(alpha=0.1)model.fit(x_poly,y)

residuals = y - model.predict(x_poly)import pylabimport statsmodels.api as smsm.qqplot(residuals, loc = 0, scale = 1, line='s')

pylab.show()

<img width="406" alt="Screen Shot 2021-03-14 at 11 34 38 AM" src="https://user-images.githubusercontent.com/74326062/111074730-c2af8b80-84ba-11eb-986b-7631b7df0647.png">

Above plot shows it almost perfectly follows a normal curve (a bit skewed at the top however)
