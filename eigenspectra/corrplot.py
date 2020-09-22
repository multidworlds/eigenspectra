#Make crosscorrelation plot showing how correlated the eigencurve coefficients are.

import numpy as np
from scipy import stats

def corrplot(file,waveindex):
	tmp=np.load(file,encoding='latin1',allow_pickle=True)
	outDictionary = tmp['arr_0'].tolist()
	eigen=outDictionary['eigencurve coefficients'][waveindex]
	corrcoeffs=-np.ones((np.shape(eigen)[1],np.shape(eigen)[1]))
	pearsonr=-np.ones((np.shape(eigen)[1],np.shape(eigen)[1]))
	for i in np.arange(np.shape(eigen)[1]):
		for j in np.arange(i+1,np.shape(eigen)[1]):
			corrcoeffs[j,i]=stats.pearsonr(eigen[:,i],eigen[:,j])[0]
			pearsonr[j,i]=stats.pearsonr(eigen[:,i],eigen[:,j])[1]
	return corrcoeffs, pearsonr
