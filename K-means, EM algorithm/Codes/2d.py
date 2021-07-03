import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import pandas as pd
from random import randrange

#Extracting dataset A2Q2_train
path = "C:\\Users\Admin\Desktop\PRML A2\A2Q2Data_test.csv"
Data = pd.read_csv(path,header=None)

Data1 = np.array(Data)
Data1 = Data1.reshape(10000,101)
X_temp = Data1[0:10000,0:100]
X_intercept = np.ones((10000,1))
X = np.hstack((X_temp,X_intercept))
Y = Data1[0:10000,100:101]
X_inv = np.linalg.inv(np.matmul(X.T,X))
W_ml = np.matmul((np.matmul(X_inv,X.T)),Y)

I = np.identity(101,dtype=float)

#Split a dataset into k folds
def cross_validation_split(dataset,folds):
    D_split = list()
    D_copy = list(dataset)
    fold_size = int(len(dataset)/folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(D_copy))
            fold.append(D_copy.pop(index))
        D_split.append(fold)
    return D_split

lambdas = 0.5
lambda_range = 50
lambdas_ridge = [0]*lambda_range
ssr_lambda = [0]*lambda_range
fold = 5
W_R = np.zeros((101,1))
for k in range(lambda_range):
    lambdas_ridge[k] = lambdas*k
    ssr = [0]*fold
    for i in range(fold):
        X_train = []
        Y_train = []
        x_folds = cross_validation_split(X,fold)
        y_folds = cross_validation_split(Y,fold)
        X_test = x_folds[i]
        Y_test = y_folds[i]
        for j in range(fold):
            if(j!=i):
                X_train.append(x_folds[j])
                Y_train.append(y_folds[j])
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_train = X_train.reshape(8000,101)
        Y_train = Y_train.reshape(8000,1)
        W_R = np.matmul(np.linalg.inv(np.matmul(X_train.T,X_train)+lambdas_ridge[k]*I),np.matmul(X_train.T,Y_train))
        Y_test_fold = np.matmul(X_test,W_R)
        ssr[i] = np.sum(np.power((Y_test_fold-Y_test),2))
    ssr_lambda[k] = np.sum(ssr)/fold
lambda_opt = np.argmin(ssr_lambda)


plt.plot(lambdas_ridge,ssr_lambda)
plt.title('Ridge Regression')
plt.xlabel('Lambda')
plt.ylabel('MSE')

W_ridge = np.matmul(np.linalg.inv(np.matmul(X.T,X)+(lambda_opt*lambdas)*I),np.matmul(X.T,Y))

#Extracting dataset A2Q2_test
path = "C:\\Users\Admin\Desktop\PRML A2\A2Q2Data_train.csv"
data_test = pd.read_csv(path,header=None)
Data1_test = np.array(data_test)
Data1_test = Data1_test.reshape(500,101)
X_temp_new = Data1_test[0:500,0:100]
X_intercept = np.ones((500,1))
X_test_data = np.hstack((X_temp_new,X_intercept))
Y_test_data = Data1_test[0:500,100:101]

Y_ml = np.matmul(X_test_data,W_ml)
Y_ridge = np.matmul(X_test_data,W_ridge)
print(np.sum(np.power((Y_test_data-Y_ml),2)),np.sum(np.power((Y_test_data-Y_ridge),2)))
plt.show()