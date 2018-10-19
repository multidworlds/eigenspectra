
"""Function to do k-means clustering on Eigenspectra
"""

from sklearn.cluster import KMeans

import numpy as np
import scipy

import sklearn
import matplotlib.pyplot as plt


from sklearn import cluster, datasets, decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets


def kmeans(fluxmap,k,labels=False):
	#Expects flux maps in the form (lam,lat,long)
	lamdim=fluxmap.shape[0]
	latdim=fluxmap.shape[1]
	longdim=fluxmap.shape[2]
	kmeans = KMeans(n_clusters=k, random_state=0).fit(fluxmap.transpose(1,2,0).reshape(-1,
		fluxmap.transpose(1,2,0).shape[2]))
	return (kmeans.labels_).reshape(latdim,longdim)



'''
n = 20
<<<<<<< HEAD
def kmeansBest(n):
    
    labels, score = np.zeros(n), np.zeros(n)
    
    for i in range(n): #try for k values from 0 to 20
        
        
        labels[i] = kmeans(fluxmap,i))
        score[i] = metrics.silhouette_score(fluxMap.reshape(), labelsForThisOne, metric='euclidean')
    
    #choose the highest score!
    k = np.where(score==np.max(score))
    labels = np.where(score==np.max(score))

=======
labels, score = np.zeros(n), np.zeros(n)

for i in range(n): #try for k values from 0 to 20

    labels[i] = kmeans(fluxmap,i)
    score[i] = metrics.silhouette_score(fluxMap.reshape(), labelsForThisOne, metric='euclidean')

#choose the highest score!
k = np.where(score==np.max(score))
labels = np.where(score==np.max(score))
'''
>>>>>>> 1fa036662d2a4f20835f7b49b3e70c0b1c0ec514


