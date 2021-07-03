import numpy as np
import csv
import numpy.linalg
import matplotlib.pyplot as plt
import pandas as pd

path = "C:\\Users\Admin\Desktop\PRML A2\A2Q2Data_test.csv"
data = pd.read_csv(path,header=None)
X1 = np.array(data)

X1 = X1.reshape(10000,101)

X_temp = X1[0:10000,0:100]
X_intercept = np.ones((10000,1))
X = np.hstack((X_temp,X_intercept))
Y = X1[0:10000,100:101]
X_inv = np.linalg.inv(np.matmul(X.T,X))
W_ml = np.matmul((np.matmul(X_inv,X.T)),Y)

print(W_ml)
wtemp = np.zeros((101,1))
W_new = []
for i in range(1,10001):
    W_l = np.sum(np.power((wtemp-W_ml),2))
    W_new.append(W_l)
    wtemp = wtemp-(np.matmul(np.matmul(X.T,X),wtemp)-np.matmul(X.T,Y))*(0.000007)
W_new1 = np.array(W_new)
W_new1 = W_new1.reshape(10000,1)
plt.plot(W_new1)
plt.title('Gradient Descent')
plt.xlabel('No of Iteration')
plt.ylabel('$||W^t-W_{ML}||_2$')
plt.show()