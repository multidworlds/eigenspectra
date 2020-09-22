"""
K-Means Clustering
------------------
Functions to do k-means clustering on Eigenspectra.
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import metrics
import pdb

def kmeans(fluxmap, k, labels=False):
    '''
    Compute k-means clustering of spectra for a given number of groups.

    Parameters
    ----------
    fluxmap : array
        Fp/Fs (axes: wavelength x latitude x longitude)
    k : int
        number of groups

    Returns
    -------
    kgroups: array of group indices (ints) from 0 to k-1 (axes: lat x lon)
    '''

	# Expects flux maps in the form (lam,lat,long)

    lamdim = fluxmap.shape[0]
    latdim = fluxmap.shape[1]
    longdim = fluxmap.shape[2]
    _fluxmap = fluxmap.transpose(1, 2, 0).reshape(-1, fluxmap.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=0).fit(_fluxmap)
    kgroups = (kmeans.labels_).reshape(latdim, longdim)
    return kgroups


def kmeansBest(fluxmap, n=10):
    '''
    Estimate the number of groups that subdivide an array of spectra.

    Parameters
    ----------
    fluxmap : array
        Fp/Fs (axes: wavelength x latitude x longitude)
    n : int
        maximum number of groups to attempt

    Returns
    -------
    k : int
        optimal number of groups
    '''

    labels, score = np.zeros(n), np.zeros(n)

    for i in range(n):  # try for k values from 0 to n
        label = kmeans(fluxmap, i)
        labels[i] = label
        score[i] = metrics.silhouette_score(fluxmap, label, metric='euclidean')

    # choose the highest score!
    k = np.argmax(score)
    return k

def sort_draws(eigenspectra_draws,kgroup_draws,uber_eigenlist,method='avg'):
    '''
    Take the many different draws and sort the groups so that we avoid
    the sorting problem where the groups are all mixed up from one
    noise instance to another

    Parameters
    ----------
    eigenspectra_draws: list or np.array
        A 3D array (or list of 2D arrays) that contain the spectra of each group

    method: str
        The name of the method to sort the eigenspectra draws
        `"avg"` - the average of each spectrum
        `"middle"` - the middle value of each spectrum

    Returns
    --------
    sortedDraws: a 3D array of spectra for each draw and group
        these will be sorted according to the `method` keyword
    '''
    eDraws = np.array(eigenspectra_draws)
    kGroup = np.array(kgroup_draws)

    if method == 'avg':
        sortValue = np.mean(eDraws,axis=2)
    elif method == 'middle':
        midWaveInd = eDraws.shape[2]/2
        middleSpec = eDraws[:,:,midWaveInd]
        sortValue = middleSpec
    else:
        print("Unrecognized sorting method")
        return 0

    sortArg = sortValue.argsort(axis=1)
    ## An ascending order array
    ascendingOrder = np.arange(sortArg.shape[1])

    sortedDraws = np.zeros_like(eDraws)
    sortedKgroups = np.zeros_like(kGroup)
    sortedubereigenlist=[[[[] for i in range(np.shape(uber_eigenlist)[2])] for i in range(np.shape(uber_eigenlist)[1])] for i in range(np.shape(uber_eigenlist)[0])]

    for ind,oneDraw in enumerate(eDraws):
        sortedDraws[ind] = oneDraw[sortArg[ind]]
        for oneGroup in ascendingOrder:
            pts = kGroup[ind] == oneGroup
            sortedKgroups[ind][pts] = sortArg[ind][oneGroup]
            for wavenum in range(np.shape(uber_eigenlist)[2]):
                sortedubereigenlist[ind][np.where(sortArg[ind]==oneGroup)[0][0]][wavenum] = uber_eigenlist[ind][oneGroup][wavenum]


    return sortedDraws, sortedKgroups,sortedubereigenlist
