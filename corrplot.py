#Make crosscorrelation plot showing how correlated the eigencurve coefficients are.

import numpy as np

def corrplot(file):
	tmp=np.load(file)
	outDictionary = tmp['arr_0'].tolist()
	eigen=outDictionary['eigencurve coefficients'][2]
	corrcoeffs=-np.ones((np.shape(eigen)[1],np.shape(eigen)[1]))
	for i in np.arange(np.shape(eigen)[1]):
		for j in np.arange(i+1,np.shape(eigen)[1]):
			corrcoeffs[j,i]=np.corrcoef(eigen[:,i],eigen[:,j])[0,1]
	return corrcoeffs
