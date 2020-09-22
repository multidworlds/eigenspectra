'''' Routines to get Eigenspectra, and lower dimensional projection, given a map in the form Spec(lam,lat,long)
 '''
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def getEigenSpec(X,n_comp):
    """
	Returns the Eigen Spectra for the dataset
    """

	#Expects flux maps in the form (lam,lat,long)
	lamdim=X.shape[0]
	latdim=X.shape[1]
	longdim=X.shape[2]
	#Flatten latitude and longitude to get 2D array, with second element a spectrum at each location
	Xreshaped=X.transpose(1,2,0).reshape(-1,X.transpose(1,2,0).shape[2])

	#Scaler to first remove mean, because PCA destroys information about mean
	scaler = StandardScaler(with_std=False,with_mean=True)
	x = scaler.fit_transform(Xreshaped)
	#Do PCA
	pca = PCA(n_components=n_comp)
	#Get coefficients. pca.components_ gives Eigenvectors.
	principalComponents = pca.fit_transform(Xreshaped)
	modelPCA_unscaled = np.dot(principalComponents,pca.components_)
	#Return eigenspectra
	return pca.components_

def getPCArepresentation(X,n_comp):
    """
	Returns map projected in lower dimensional spectral space using Eigen Spectra
    """

	#Expects flux maps in the form (lam,lat,long)
	lamdim=X.shape[0]
	latdim=X.shape[1]
	longdim=X.shape[2]
	#Flatten latitude and longitude to get 2D array, with second element a spectrum at each location
	Xreshaped=X.transpose(1,2,0).reshape(-1,X.transpose(1,2,0).shape[2])

	#Scaler to first remove mean, because PCA destroys information about mean
	scaler = StandardScaler(with_std=False,with_mean=True)
	x = scaler.fit_transform(Xreshaped)
	#Do PCA
	pca = PCA(n_components=n_comp)
	#Get coefficients. pca.components_ gives Eigenvectors.
	principalComponents = pca.fit_transform(Xreshaped)
	modelPCA_unscaled = np.dot(principalComponents,pca.components_)
	fullModel = scaler.inverse_transform(modelPCA_unscaled)
	#Reshape model to Spectrum(lam,lat,long)
	return fullModel.transpose(1,0).reshape(lamdim,latdim,longdim)
