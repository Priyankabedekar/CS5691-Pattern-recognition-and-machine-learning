import numpy as np
import csv
import matplotlib.pyplot as plt
import random

x = []

with open('A2Q1Data.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    for row in plots:
        x.append(int(row[0]))

X = np.array(x)
X = X.reshape((1000,1))
N = 1000

K  = 4

np.random.seed(0)
Lambda = np.zeros((N,K))
sig=np.zeros((4,1))

logLL = np.zeros((30,1))
for I in range(0,100):
    mu = np.random.default_rng().uniform(0,1,K)
    sig=[0.1,0.1,0.1,0.1]
    Pi = np.random.default_rng().uniform(0,1,K)
    Pi /= Pi.sum()
    # print(Pi)
    for T in range(30):
        Temp_P = np.zeros((K,1))
        for i in range(N):
            denom = 0
            for l in range(K):
                sigma = 2 * sig[l]
                denom += np.exp((-1 * (X[i] - mu[l])*(X[i] - mu[l]))/sigma) * Pi[l]
            for k in range(K):
                sigma = 2 * sig[k]
                Lambda[i][k] = (np.exp((-1 * (X[i] - mu[k])*(X[i] - mu[k]))/sigma) * Pi[k]) / denom
                logLL[T] +=  Lambda[i][k] * (np.log(Pi[k]) - (((X[i] - mu[k])*(X[i] - mu[k]))/sigma) - np.log(Lambda[i][k]))
            Pi = Lambda.sum(axis = 0)
            for k in range(K):
               Temp_P[k] += Lambda[i][k] * X[i]
        for z in range (K):
            mu[z] = Temp_P[z] / Pi[z]
        Pi = Pi/N
logLL = logLL / 100

# for i in range(30):
#     plt.scatter(i,logLL[i])
# plt.title('GMM')
# plt.xlabel('No of Iterations')
# plt.ylabel('Log likelihood')
# plt.show()  

plt.plot(logLL)
plt.title('GMM')
plt.xlabel('No of Iterations')
plt.ylabel('Log likelihood')
plt.show()
