import numpy as np
import csv
from numpy import linalg as LA
import matplotlib.pyplot as plt
x = []
y = []

with open('Dataset3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))


plt.scatter(x,y, s=None)
plt.title('Original data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

Size = len(x)

#covariance matrix
Z = np.array([x,y])
Cov = (1/Size) * np.dot(Z,Z.T)

#Calculate Eigenvalues and vectors
Eigvalues, Eigvectors = LA.eig(Cov)

print(Eigvalues)
print(Eigvectors)

index = Eigvalues.argsort()[:: -1]
Eigvalues = Eigvalues[index]
Eigvectors = Eigvectors[:,index]

print(Eigvalues)
print(Eigvectors)
W = Eigvectors
#project data on to principle components

Zi = np.dot(W.T,Z)

plt.scatter(Zi[0],Zi[1],color = 'g',marker='*')
plt.title('plotting after PCA without centering')
plt.xlabel('principle component 1')
plt.ylabel('principle component 2')
plt.show()

#Variance given by each Principle component
Variance = (1/Size) * ((np.dot(W.T,Z))**2)

PC1 = sum(Variance[0])/sum(Variance[0]+Variance[1])
PC2 = sum(Variance[1])/sum(Variance[0]+Variance[1])
print(PC1,PC2)

print("Variance from first principle component" ,PC1 *100 ,"%")
print("Variance from second principle component" ,PC2 *100,"%")


