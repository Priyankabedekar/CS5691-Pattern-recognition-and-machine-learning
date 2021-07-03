import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import pandas as pd

path = "C:\\Users\Admin\Desktop\PRML A2\A2Q2Data_test.csv"

Data = pd.read_csv(path,header=None)
Data1 = np.array(Data)
Data1 = Data1.reshape(10000,101)
X_temp = Data1[0:10000,0:100]
X_intercept = np.ones((10000,1))
X = np.hstack((X_temp,X_intercept))
Y = Data1[0:10000,100:101]
X_inv = np.linalg.inv(np.matmul(X.T,X))
W_ML = np.matmul((np.matmul(X_inv,X.T)),Y)

print(W_ML)