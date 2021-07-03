from PIL import Image 
import matplotlib.pyplot as plt
import csv
import numpy as np
import numpy.linalg as LA
from os import listdir
from matplotlib import image


Total = 40
size = 50 #size of image

images = []

#Read data from category1 and category2
for filename in listdir('Dataset/Category1/'):
    image = Image.open('Dataset/Category1/' + filename)
    im = np.array((image.resize((size,size))).convert('L'))
    row, col = im.shape
    i = im.reshape(row*col)
    images.append(i)
for filename in listdir('Dataset/Category2/'):
    image = Image.open('Dataset/Category2/' + filename)
    im = np.array((image.resize((size,size))).convert('L'))
    row, col = im.shape
    i = im.reshape(row*col)
    images.append(i)

X = np.array(images).T  #columns are flatten images i.e features

meanX = np.mean(X,axis=1,keepdims=True)
inputMatrix = X-meanX

covariance =  (np.dot(inputMatrix ,inputMatrix.T) )/float(Total)

Eigvalues, Eigvectors = LA.eigh(covariance)

index = Eigvalues.argsort()[:: -1]
Eigvalues = Eigvalues[index]
Eigvectors = Eigvectors[:,index]

Eigvectors = Eigvectors.T


c,principalComponents = Eigvectors.shape
componetPercentage = [.10,.20,.30,.50,.75,1]


projections = np.dot(Eigvectors,inputMatrix)  #projections of input on eigen vectors

kVariousComponents = []
for percent in componetPercentage:
    kComponents = int(principalComponents*percent)
    projected = np.dot(Eigvectors.T[:,0:kComponents],projections[0:kComponents,:])
    kVariousComponents.append(projected)

kVariousComponents = np.array(kVariousComponents)

'''
for i in range(0,6):
    kVariousComponents[i] = kVariousComponents[i]+meanX
    for j in range(0,40):
        a = 'output/' + str(i)+'_'+str(j) + '.jpg'
        #str a = "{}{}{}".format('output/', j,'.jpg')
        out = kVariousComponents[i][:,j].reshape((size,size))
        Image.fromarray(out).convert("L").save(a)
'''
#Reading test images 
testImages = []
for filename in listdir('Dataset/test'):
    image = Image.open('Dataset/test/' + filename)
    im = np.array((image.resize((size,size))).convert('L'))
    row, col = im.shape
    i = im.reshape(row*col)
    testImages.append(i)
testX = np.array(testImages).T
testX = testX-meanX

projections = np.dot(Eigvectors,testX)

testKVariousComponents = []
for percent in componetPercentage:
    kComponents = int(principalComponents*percent)
    projected = np.dot(Eigvectors.T[:,0:kComponents+1],projections[0:kComponents+1,:])
    testKVariousComponents.append(projected)


testKVariousComponents = np.array(testKVariousComponents)

#Output test images with various principal components

""" for i in range(0,6):
    testKVariousComponents[i] = testKVariousComponents[i]+meanX
    for j in range(0,10):
        a = 'testOutput/' + str(i) + '_' + str(j) + '.jpg'
        #str a = "{}{}{}".format('output/', j,'.jpg')
        out = testKVariousComponents[i][:,j].reshape((size,size))
        Image.fromarray(out).convert("L").save(a)

 """
category1 = []
category2 = []

for k in range(0,6):
    tempCat1 = []
    tempCat2 = []

    for i in range(0,10):    
        a,b = testKVariousComponents[k].shape
        a = np.sqrt(np.sum(np.multiply(kVariousComponents[k]-testKVariousComponents[k][:,i].reshape((a,1)),kVariousComponents[k]-testKVariousComponents[k][:,i].reshape((a,1))),axis=0))
        cat1 = np.sum(a[0:19])/20
        cat2 = np.sum(a[20:39])/20
        tempCat1.append(cat1)
        tempCat2.append(cat2)

    category1.append(tempCat1)
    category2.append(tempCat2)

category1 = np.array(category1)
category2 = np.array(category2)
#prints average distance of each test images with category 1 and category2 
#Each row represents different value of principal components taken into account
#Column represents no of test cases and each entry corresponds to average distance from category
print("Avg. distance from category1 : ",category1)
print("Avg. distance from category1 : ",category2)

#Predicting Test images
Predicted = []
for i in range(0,6):
    temp = []
    for j in range(0,10):
        if category1[i][j] < category2[i][j]:
            temp.append("Cat1")
        elif category1[i][j] > category2[i][j]:
            temp.append("Cat2")
        else:
            temp.append("Got Confused")
    Predicted.append(temp)
print(Predicted)