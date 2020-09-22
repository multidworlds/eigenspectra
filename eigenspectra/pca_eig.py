# PCA routine from http://glowingpython.blogspot.sg/2011/07/principal-component-analysis-with-numpy.html
# see also: http://glowingpython.blogspot.it/2011/07/pca-and-image-compression-with-numpy.html
# fyi, python arrays are [rows,columns]

from numpy import mean,cov,cumsum,dot,linalg,size,flipud,argsort

def princomp(A):
  """ performs principal components analysis 
      (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to variables. 

     Returns :  
      coeff :
        is a p-by-p matrix, each column containing coefficients 
        for one principal component.
      score : 
        the principal component scores; that is, the representation 
        of A in the principal component space. Rows of SCORE 
        correspond to observations, columns to components.
      latent : 
        a vector containing the eigenvalues 
        of the covariance matrix of A.
    """
  # computing eigenvalues and eigenvectors of covariance matrix
  M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
  [latent,coeff] = linalg.eig(cov(M))
  idx = argsort(latent) # sorting the eigenvalues
  idx = idx[::-1]       # in ascending order
  # sorting eigenvectors according to the sorted eigenvalues
  coeff = coeff[:,idx]
  latent = latent[idx] # sorting eigenvalues
  score = dot(coeff.T,M) # projection of the data in the new space
  return coeff,score,latent