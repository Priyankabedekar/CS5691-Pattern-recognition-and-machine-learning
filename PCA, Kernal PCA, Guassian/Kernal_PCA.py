import numpy as np
import csv
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt
x = []
y = []

with open('Dataset3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))

Size = len(x)

plt.scatter(x,y, s=None)
plt.title('Original data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

X = np.array([x,y])
# Kernal matrix for (1 + x^Ty)^2

K = (np.dot(X.T,X) + 1)*(np.dot(X.T,X) + 1)

#center K
N = np.full((Size , Size), 1/Size)
I = np.identity(Size)
Kcentered = (I - N) * K * (I - N)

#Eigebvectors
Eigvalues , Eigvectors = LA.eig(K)
index = Eigvalues.argsort()[:: -1]
Eigvalues = Eigvalues[index]
Eigvectors = Eigvectors[:,index]

Top_2_EV = Eigvectors[:,[0,1]]
#print(Eigvalues[0],Eigvalues[1])
print(Eigvectors[:,0],Eigvectors[:,1])


plt.scatter(Eigvectors[:,0],Eigvectors[:,1],s=None)
plt.title('plotting after polynomial kernal (d = 2)')
plt.xlabel('principle component 1')
plt.ylabel('principle component 2')
plt.show()


#Kernal matrix for d=3
K3 = (np.dot(X.T,X) + 1)*(np.dot(X.T,X) + 1)*(np.dot(X.T,X) + 1)

#Eigebvectors
Eigvalues1 , Eigvectors1 = LA.eig(K3)
index = Eigvalues1.argsort()[:: -1]
Eigvalues1 = Eigvalues1[index]
Eigvectors1 = Eigvectors1[:,index]


plt.scatter(Eigvectors1[:,0],Eigvectors1[:,1],s=None)
plt.title('plotting after polynomial kernal (d = 3)')
plt.xlabel('principle component 1')
plt.ylabel('principle component 2')
plt.show()


