
"""Function to do k-means clustering on Eigenspectra
"""


def(fluxmap,k,labels=False):
	#Expects flux maps in the form (lam,lat,long)
	lamdim=fluxmap.shape[0]
	latdim=fluxmap.shape[1]
	longdim=fluxmap.shape[2]
	kmeans = KMeans(n_clusters=k, random_state=0).fit(fluxmap.transpose(1,2,0).reshape(-1,
		fluxmap.transpose(1,2,0).shape[2]))
	return (kmeans.labels_).reshape(latdim,longdim)