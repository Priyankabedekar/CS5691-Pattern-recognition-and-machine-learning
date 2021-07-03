import numpy as np
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

N = 100

X_S = np.zeros((N,101))
Y_S = np.zeros((N,1))
W_temp_S = np.zeros((101,1))
W_S = []

for i in range(10000):
    for j in range(N):
        k = np.random.randint(0,9999)
        X_S[j,:] = X[k,:]
        Y_S[j] = Y[k]
    W_l2 = np.sum(np.power((W_temp_S-W_ml),2))
    W_S.append(W_l2)
    W_temp_S=W_temp_S-(np.matmul(np.matmul(X_S.T,X_S),W_temp_S)-np.matmul(X_S.T,Y_S))*(0.0005)
W_S=np.array(W_S)

W_S=W_S.reshape(10000,1)


plt.plot(W_S)
plt.title('Stochastic Gradient Descent')
plt.xlabel('No of Iteration')
plt.ylabel('$||W^t-W_{ML}||_2$')
plt.show()