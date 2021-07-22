#!/usr/bin/env python
# coding: utf-8


# 
# (1) Use read_csv() to load and examine each dataset.
# (2) Use logistic regression to fit() and score() a binary classifier for dataset 1. How accurate are the model’s predictions?
# (3) Repeat experiment (2) for dataset 2. How wellclassifiers does it score?
# (4) Create separate scatterplots for datasets 1 and 2, plotting points from class 0 with a different color and marker from points in class 1. What accounts for the discrepancies between experiments (2) and (3)?
# (5) Fit and score Gaussian Naive Bayes classifiers for datasets 1 and 2. How well do these classifiers score compared to logistic regression?
# (6) Repeat experiment (5) with K-Nearest Neighbor classifiers.
# (7) Use the code from KV Subbaiah Setty’s tutorial How To Plot A Decision Boundary For Machine Learning Algorithms in Python as a guide, plot the decision boundaries for each classifier and dataset. What differences do you observe?
# (8) Now repeat experiments (2), (5), (6), and (7) with dataset 3.  
# 

# In[284]:


import numpy as np 
import matplotlib.pyplot as plt
import sklearn.model_selection as sk 
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# In[285]:


# (1) Use read_csv() to load and examine each dataset.
data1 = pd.read_csv('dataset1.csv')
data2 = pd.read_csv('dataset2.csv')
data3 = pd.read_csv('dataset3.csv')
print(data1.keys())
print(data2.keys())
print(data3.keys())
#store keyvalues for later use
data1x1 = data1.keys()[0]
data1x2 = data1.keys()[1]
data2x1 = data2.keys()[0]
data2x2 = data2.keys()[1]
data3x1 = data3.keys()[0]
data3x2 = data3.keys()[1]


# In[286]:


# (2) Use logistic regression to fit() and score() a binary classifier for dataset 1. How accurate are the model’s predictions?
#  fit(X, y, sample_weight=None)[source]


data1x = data1[['1.8005387353055742','-0.5392180610512733']]
data1t = data1['0']
data2x = data2[['-1.6644070030229474','17.24541535537469']]
data2t = data2['0']
data3x = data3[['30.876149283835716','7.806792944703234']]
data3t = data3['0']
#print(data1x)

regressor = LogisticRegression() 
regressor.fit(data1x, data1t)
print(regressor.score(data1x, data1t))
# print(regressor.score(data2x, data2t)) 0.49246231155778897
# print(regressor.score(data3x, data3t)) 0.5025125628140703
#score for data1 is 1, the model appears to be accurate

# (3) Repeat experiment (2) for dataset 2. How well does it score?
regressor2 = LogisticRegression()
regressor2.fit(data2x, data2t)
print(regressor2.score(data2x, data2t))
#regressor2.score() is 0.542713567839196, not as good as regressor1 


# In[287]:


# (4) Create separate scatterplots for datasets 1 and 2, plotting points from class 0 with a different color and marker from points in class 1. 
# What accounts for the discrepancies between experiments (2) and (3)?

data1.plot.scatter(data1x1,data1x2, c= '0', colormap='jet')
data2.plot.scatter(data2x1,data2x2, c= '0', colormap='jet')

#The classifications are different in the manner that in experiment (2) 
# the data1 data can be seperated between classifications with a line
#  where as the data from data2 is concentric classifation. See below scatter plots. 


# In[288]:


# (5) Fit and score Gaussian Naive Bayes classifiers for datasets 1 and 2. 
#  How well do these classifiers score compared to logistic regression?

gnb1 = GaussianNB()
gnb1.fit(data1x, data2t)
print(gnb1.score(data1x,data1t)) #score 0.010050251256281407 the model performed very poorly 

gnb2 = GaussianNB()
gnb2.fit(data2x, data2t)
print(gnb1.score(data2x,data2t)) #score 0.49246231155778897  the model performed worse than the logistic model 

# (6) Repeat experiment (5) with K-Nearest Neighbor classifiers.
knn1 = KNeighborsClassifier(n_neighbors=3)
knn1.fit(data1x,data1t)
print(knn1.score(data1x,data1t)) #score of 1, the model fits the data perfectly

knn2 = KNeighborsClassifier(n_neighbors=3)
knn2.fit(data2x,data2t)
print(knn2.score(data2x,data2t)) #score of 1, the model fits the data perfectly


# In[289]:


#(7) Use the code from KV Subbaiah Setty’s tutorial How To Plot A Decision Boundary For Machine Learning Algorithms in Python as a guide, plot the decision boundaries for each classifier and dataset. What differences do you observe?

# define bounds of the domain
data1_min1, data1_max1 = data1x[data1x1].min()-1, data1x[data1x1].max()+1
data1_min2, data1_max2 = data1x[data1x2].min()-1, data1x[data1x2].max()+1
data2_min1, data2_max1 = data2x[data2x1].min()-1, data2x[data2x1].max()+1
data2_min2, data2_max2 = data2x[data2x2].min()-1, data2x[data2x2].max()+1

# define the x and y scale
data1_x1grid = np.arange(data1_min1, data1_max1, 0.1)
data1_x2grid = np.arange(data1_min2, data1_max2, 0.1)
data2_x1grid = np.arange(data2_min1, data2_max1, 0.1)
data2_x2grid = np.arange(data2_min2, data2_max2, 0.1)

# create all of the lines and rows of the grid
data1_xx, data1_yy = np.meshgrid(data1_x1grid, data1_x2grid)
data2_xx, data2_yy = np.meshgrid(data2_x1grid, data2_x2grid)


# flatten each grid to a vector
data1_r1, data1_r2 = data1_xx.flatten(), data1_yy.flatten()
data1_r1, data1_r2 = data1_r1.reshape((len(data1_r1), 1)), data1_r2.reshape((len(data1_r2), 1))
data2_r1, data2_r2 = data2_xx.flatten(), data2_yy.flatten()
data2_r1, data2_r2 = data2_r1.reshape((len(data2_r1), 1)), data2_r2.reshape((len(data2_r2), 1))

# horizontal stack vectors to create x1,x2 input for the model
data1_grid = np.hstack((data1_r1,data1_r2))
data2_grid = np.hstack((data2_r1,data2_r2))

# make predictions for the grid
model1 = LogisticRegression()
model2 = LogisticRegression()
data1_yhat = regressor.predict(data1_grid)
data2_yhat = regressor2.predict(data2_grid)


# reshape the predictions back into a grid
data1_zz = data1_yhat.reshape(data1_xx.shape)
data2_zz = data2_yhat.reshape(data2_xx.shape)

# plot the grid of x, y and z values as a surface
plt.contourf(data1_xx, data1_yy, data1_zz, cmap='jet')
for class_value in range(2):
    row_ix = np.where(data1t == class_value)
    plt.scatter(data1[data1x1], data1[data1x2], c= '0', cmap='jet')


# In[290]:


plt.contourf(data2_xx, data2_yy, data2_zz, cmap='jet')
for class_value in range(2):
    plt.scatter(data2[data2x1], data2[data2x2], cmap='jet')


# In[291]:


#(8) Now repeat experiments (2), (5), (6), and (7) with dataset 3.  

#(2)
regressor3 = LogisticRegression() 
regressor3.fit(data1x, data1t)
print(regressor3.score(data3x, data3t)) 
# score 0.5025125628140703, the model is as accurate as random selection
#(5)
gnb3 = GaussianNB()
gnb3.fit(data3x, data3t)
print(gnb3.score(data3x,data3t)) 
#score 0.8844221105527639, the model is pretty accurate. 
#(6)
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(data3x,data3t)
print(knn3.score(data3x,data3t)) 
#score 1.0, the model fits perfectly. with all three data sets the KNN model was the best fitting 

#(7) Use the code from KV Subbaiah Setty’s tutorial How To Plot A Decision Boundary For Machine Learning Algorithms in Python as a guide, plot the decision boundaries for each classifier and dataset. What differences do you observe?

# define bounds of the domain
data3_min1, data3_max1 = data3x[data3x1].min()-1, data3x[data3x1].max()+1
data3_min2, data3_max2 = data3x[data3x2].min()-1, data3x[data3x2].max()+1


data3_min1, data3_max1 = data3x[data3x1].min()-1, data3x[data3x1].max()+1
data3_min2, data3_max2 = data3x[data3x2].min()-1, data3x[data3x2].max()+1


# define the x and y scale
data3_x1grid = np.arange(data3_min1, data3_max1, 0.1)
data3_x2grid = np.arange(data3_min2, data3_max2, 0.1)

# create all of the lines and rows of the grid
data3_xx, data3_yy = np.meshgrid(data3_x1grid, data3_x2grid)

# flatten each grid to a vector
data3_r1, data3_r2 = data3_xx.flatten(), data3_yy.flatten()
data3_r1, data3_r2 = data3_r1.reshape((len(data3_r1), 1)), data3_r2.reshape((len(data3_r2), 1))

# horizontal stack vectors to create x1,x2 input for the model
data3_grid = np.hstack((data3_r1,data3_r2))

# make predictions for the grid
model1 = LogisticRegression()
data3_yhat = regressor.predict(data3_grid)

# reshape the predictions back into a grid
data3_zz = data3_yhat.reshape(data3_xx.shape)

# plot the grid of x, y and z values as a surface
plt.contourf(data3_xx, data3_yy, data3_zz, cmap='jet')
for class_value in range(2):
    #row_ix = np.where(y == class_value)
    plt.scatter(data3[data3x1], data3[data3x2], cmap='jet')


# In[ ]:




