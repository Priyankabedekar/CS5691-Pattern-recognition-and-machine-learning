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
plt.show()

X = np.array([y,x])
print(X.shape)

#Gaussian kernal 

fig = plt.figure(figsize=(3,3))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3  = fig.add_subplot(133)
fig2 = plt.figure(figsize=(3,3))
ax4 = fig2.add_subplot(131)
ax5 = fig2.add_subplot(132)
ax6 = fig2.add_subplot(133)
fig3 = plt.figure(figsize=(3,3))
ax7 = fig3.add_subplot(141)
ax8 = fig3.add_subplot(142)
ax9 = fig3.add_subplot(143)
ax10 = fig3.add_subplot(144)



def GaussianMatrix(X,sigma):
    row,col = X.shape
    GassMatrix = np.zeros(shape=(row,row))
    X=np.asarray(X)
    i=0
    for i1 in X:
        j=0
        for j1 in X:
            GassMatrix[i,j]=Gaussian(i1.T,j1.T,sigma)
            j+=1
        i+=1
    return GassMatrix
def Gaussian(x,z,sigma):
    return np.exp((-(np.linalg.norm(x-z)**2))/(2*sigma**2))

 #Eigebvectors
def Cal_Kernal(K):
    Eigvalues2 , Eigvectors2 = LA.eig(K)
    index = Eigvalues2.argsort()[:: -1]
    Eigvalues2 = Eigvalues2[index]
    Eigvectors2 = Eigvectors2[:,index]


    Scale1 = math.sqrt(1/(1000 * Eigvalues2[0]))
    Scale2 = math.sqrt(1/(1000 * Eigvalues2[1]))

    E1 = []
    E2 = []

    E1 = Scale1 * Eigvectors2[:,0]
    E2 = Scale2 * Eigvectors2[:,1]

    E = np.array([E1,E2])

    Zi = np.dot(E.T,X)
    return Zi

K = GaussianMatrix(X.T,0.1)
Zi = Cal_Kernal(K)
ax1.scatter(Zi[0],Zi[1],color = 'g',marker='.')
ax1.set_title('sigma = 0.1')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

K = GaussianMatrix(X.T,0.2)
Zi = Cal_Kernal(K)
ax2.scatter(Zi[0],Zi[1],color = 'g',marker='.')
ax2.set_title('sigma = 0.2')
ax2.set_xlabel('x')
ax2.set_ylabel('y')


K = GaussianMatrix(X.T,0.3)
Zi = Cal_Kernal(K)
ax3.scatter(Zi[0],Zi[1],color = 'g',marker='.')
ax3.set_title('sigma = 0.3')
ax3.set_xlabel('x')
ax3.set_ylabel('y')


K = GaussianMatrix(X.T,0.4)
Zi = Cal_Kernal(K)
ax4.scatter(Zi[0],Zi[1],color = 'g',marker='.')
ax4.set_title('sigma = 0.4')
ax4.set_xlabel('x')
ax4.set_ylabel('y')

K = GaussianMatrix(X.T,0.5)
Zi = Cal_Kernal(K)
ax5.scatter(Zi[0],Zi[1],color = 'g',marker='.')
ax5.set_title('sigma = 0.5')
ax5.set_xlabel('x')
ax5.set_ylabel('y')



K = GaussianMatrix(X.T,0.6)
Zi = Cal_Kernal(K)
ax6.scatter(Zi[0],Zi[1],color = 'g',marker='.')
ax6.set_title('sigma = 0.6')
ax6.set_xlabel('x')
ax6.set_ylabel('y')

K = GaussianMatrix(X.T,0.7)
Zi = Cal_Kernal(K)
ax7.scatter(Zi[0],Zi[1],color = 'g',marker='.')
ax7.set_title('sigma = 0.7')
ax7.set_xlabel('x')
ax7.set_ylabel('y')

K = GaussianMatrix(X.T,0.8)
Zi = Cal_Kernal(K)
ax8.scatter(Zi[0],Zi[1],color = 'g',marker='.')
ax8.set_title('sigma = 0.8')
ax8.set_xlabel('x')
ax8.set_ylabel('y')


K = GaussianMatrix(X.T,0.9)
Zi = Cal_Kernal(K)
ax9.scatter(Zi[0],Zi[1],color = 'g',marker='.')
ax9.set_title('sigma = 0.9')
ax9.set_xlabel('x')
ax9.set_ylabel('y')

K = GaussianMatrix(X.T,1.)
Zi = Cal_Kernal(K)
ax10.scatter(Zi[0],Zi[1],color = 'g',marker='.')
ax10.set_title('sigma = 1.')
ax10.set_xlabel('x')
ax10.set_ylabel('y')

plt.show()

