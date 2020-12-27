import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import seaborn as sns
import os, re
from pprint import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

%matplotlib inline
testing_df = pd.read_csv('optdigits.xlsx',header=None)
X_testing,  y_testing  = testing_df.loc[:,0:63],  testing_df.loc[:,64]

training_df = pd.read_csv('optdigits.xlsx',header=None)
X_training, y_training = training_df.loc[:,0:63], training_df.loc[:,64]

#Size and Shape of data
print X_training.shape
print y_training.shape

#display data
mat = X_training.loc[0,:].reshape(8,8)
print mat
plt.imshow(mat)
gca().grid(False)
plt.show()


#normalize data
def get_normed_mean_cov(X):
    X_std = StandardScaler().fit_transform(X)
    X_mean = np.mean(X_std, axis=0)
    
    ## Automatic:
    #X_cov = np.cov(X_std.T)
    
    # Manual:
    X_cov = (X_std - X_mean).T.dot((X_std - X_mean)) / (X_std.shape[0]-1)
    
    return X_std, X_mean, X_cov

X_std, X_mean, X_cov = get_normed_mean_cov(X_training)
X_std_validation, _, _ = get_normed_mean_cov(X_testing)

#PCA Classifier
pca2 = PCA(n_components=2, whiten=True)
pca2.fit(X_std)
X_red = pca2.transform(X_std)

# State Vector Classifier
linclass2 = SVC()
linclass2.fit(X_red,y_training)

def get_cmap(n):
    #colorz = plt.cm.cool
    colorz = plt.get_cmap('Set1')
    return [ colorz(float(i)/n) for i in range(n)]

colorz = get_cmap(10)
colors = [colorz[yy] for yy in y_training]

scatter(X_red[:,0], X_red[:,1], 
        color=colors, marker='*')
xlabel("Principal Component 0")
ylabel("Principal Component 1")
title("Handwritten Digit Data in\nPrincipal Component Space",size=14)
show()