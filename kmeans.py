
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
    '''
    Compute k-mean clustering of spectra for a given number of groups.

   
    Parameters
    ----------
    fluxmap : array of Fp/Fs (axes: wavelength x latitude x longitude)
    k : int, number of groups

    Returns
    -------
    kgroups: array of group indices (ints) from 0 to k-1 (axes: lat x lon)
    '''


	#Expects flux maps in the form (lam,lat,long)
	lamdim=fluxmap.shape[0]
	latdim=fluxmap.shape[1]
	longdim=fluxmap.shape[2]
	kmeans = KMeans(n_clusters=k, random_state=0).fit(fluxmap.transpose(1,2,0).reshape(-1,
		fluxmap.transpose(1,2,0).shape[2]))
    kgroups = (kmeans.labels_).reshape(latdim,longdim)
	return kgroups




def kmeansBest(fluxmap, n=10):
    '''
    Estimate the number of groups that subdivide an array of spectra.

    Parameters
    ----------
    fluxmap : array of Fp/Fs (axes: wavelength x latitude x longitude)
    n : int, maximum number of groups to attempt

    Returns
    -------
    k : int, optimal number of groups
    '''

    labels, score = np.zeros(n), np.zeros(n)
    
    for i in range(n): #try for k values from 0 to n
        labels[i] = kmeans(fluxmap,i)
        score[i] = metrics.silhouette_score(fluxMap.reshape(), labelsForThisOne, metric='euclidean')
    
    #choose the highest score!
    k = np.argmax(score)
    return k

