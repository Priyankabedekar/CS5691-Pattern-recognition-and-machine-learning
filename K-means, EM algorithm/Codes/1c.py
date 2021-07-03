import numpy as np
import csv
import matplotlib.pyplot as plt
x = []

with open('A2Q1Data.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    for row in plots:
        x.append(int(row[0]))

X = np.array(x)
X = X.reshape((1000,1))
N = 1000
K = 4
#np.random.seed(5) 
cluster = np.random.randint(0,4,N,dtype=int)
new_cluster = np.zeros((N,1),dtype=int)
cluster_count = np.zeros((K,1))
Means = np.zeros((K,1))
objective = []
print(cluster)
t = 0
for I in range(1000):
    cluster_count = np.zeros((K,1))
    Means = np.zeros((K,1))
    for i in range(N):
        cluster_count[cluster[i]] += 1
        Means[cluster[i]] += X[i]
    
    print(Means, cluster_count)
    
    for k in range(K):
        Means[k] /= cluster_count[k]

    sum = 0
    for i in range(N):
        sum += pow(X[i] - Means[cluster[i]] , 2)
    objective.append(sum)

    for i in range(N):
        new_cluster[i] = np.argmin(np.multiply(Means-X[i],Means-X[i]))
 
    if ((new_cluster == cluster).all()):
        break
    cluster = new_cluster
    t+=1
    print("Iteration : ",t)

print(Means, cluster_count)
plt.plot(objective)
#plt.axis([0, noOfIterations,0, 1000])
plt.title('K-Means Objective')
plt.xlabel('No of Iterations')
plt.ylabel('MSE')
plt.show()

