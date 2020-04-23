#Function to fit a map at each wavelength using Eigencurves

#THINGS THAT WILL BE UPDATED IN FUTURE VERSIONS:
#	1. Right now it just loads in a single file, and for the entire pipeline it will load in a file for each wavelength and run the fit for each wavelength
#	2. It will eventually output wavelengths in addition to the spherical harmonic coefficients
#	3. Right now it just selects a number of eigencurves to include. Will eventually optimize this from chi-squared.

#INPUTS:
#	Secondary eclipse light curves at each wavelength (csv for each wavelength)

#OUTPUTS:
#	Coefficients for each of the spherical harmonics
#from lightcurves_sh_starry import sh_lcs
from lightcurves_sh import sh_lcs
from pca_eig import princomp
import numpy as np 
import matplotlib.pyplot as plt 
import emcee
import csv
import spiderman as sp
from scipy.optimize import leastsq
import pdb
from scipy import stats
from scipy import special

def mpmodel(p,x,y,z,elc,escore,nparams):#fjac=None,x=None,y=None,err=None):
	model = p[0]*elc[0,:] + p[1]
	for ind in range(2,nparams): #FINDME changed to go to nparams+1
		model = model + p[ind] * escore[ind-2,:]
	return np.array(y-model)

def lnprob(theta,x,y,yerr,elc,escore,nparams):
	lp=lnprior(theta,nparams)
	return lp+lnlike(theta,x,y,yerr,elc,escore,nparams)

def lnlike(theta,x,y,yerr,elc,escore,nparams):
	model = theta[0] * elc[0,:] + theta[1]
	for ind in range(2,nparams): #FINDME changed to go to nparams+1
		model = model + theta[ind] * escore[ind-2,:]
	resid=y-model
	chi2=np.sum((resid/yerr)**2)
	dof=np.shape(y)[0]-1.
	chi2red=chi2/dof
	ln_likelihood=-0.5*(np.sum((resid/yerr)**2 + np.log(2.0*np.pi*(yerr)**2)))
	return ln_likelihood

def lnprior(theta,nparams):
	lnpriorprob=0.
	c0 = theta[0]
	fstar = theta[1]
	if fstar<0.:
		lnpriorprob=-np.inf
	elif c0<0.:
		lnpriorprob=-np.inf
	return lnpriorprob

def eigencurves(dict,plot=False,degree=3,afew=5):
	waves=dict['wavelength (um)']
	times=dict['time (days)']
	fluxes=dict['flux (ppm)']	#2D array times, waves
	errors=dict['flux err (ppm)']
	
	burnin=200
	nsteps=1000
	nwalkers=32 #number of walkers	
	
	
	#This file is going to be a 3D numpy array 
	## The shape is (nsamples,n parameters,n waves)
	## where nsamples is the number of posterior MCMC samples
	## n parameters is the number of parameters
	## and n waves is the number of wavelengths looped over
	
	alltheoutput=np.zeros(((nsteps-burnin)*nwalkers,int((degree)**2.),np.shape(waves)[0]))
	bestfitoutput=np.zeros((int((degree)**2.),np.shape(waves)[0]))
	
	if np.shape(fluxes)[0]==np.shape(waves)[0]:
		rows=True
	elif np.shape(fluxes)[0]==np.shape(times)[0]:
		rows=False
	else:
		assert (np.shape(fluxes)[0]==np.shape(times)[0]) | (np.shape(fluxes)[0]==np.shape(waves)[0]),"Flux array dimension must match wavelength and time arrays."

	nParamsUsed, ecoeffList, escoreList,elatentList = [], [], [], []
	#elcList = []
	eigencurvecoeffList = []
	for counter in np.arange(np.shape(waves)[0]):
		wavelength=waves[counter] #wavelength this secondary eclipse is for
		eclipsetimes=times	#in days
		if rows:
			eclipsefluxes=fluxes[counter,:]*10.**-6.
			eclipseerrors=errors[counter,:]*10.**-6.
		else:
			eclipsefluxes=fluxes[:,counter]*10.**-6.
			eclipseerrors=errors[:,counter]*10.**-6.

		#alltheoutput[counter,0]=wavelength

		#	Calculate spherical harmonic maps using SPIDERMAN
		
		lc,t = sh_lcs(t0=-2.21857/2.,ntimes=eclipsetimes,degree=degree)#degree=lmax+1)#	#model it for the times of the observations
		# for i in range(16):
		# 	plt.figure()
		# 	plt.plot(t,lc[i,:])
		# 	plt.show()
		# pdb.set_trace()
		#print(np.shape(lc),np.shape(t))
		#Optional add-on above: test whether we want to include higher-order spherical harmonics when making our eigencurves?

		# subtract off stellar flux
		#lc = lc-1

		# just analyze time around secondary eclipse (from start to end of observations)
		starttime=np.min(eclipsetimes)
		endtime=np.max(eclipsetimes)
		ok=np.where((t>=starttime) & (t<=endtime))
		et = t[ok]
		elc=np.zeros((np.shape(lc)[0],np.shape(ok)[1]))
		for i in np.arange(np.shape(lc)[0]):
			elc[i,:] = lc[i,ok]

		#  PCA
		ecoeff,escore,elatent = princomp(elc[1:,:].T)
		escore=np.real(escore)

		if isinstance(afew,list):#np.shape(afew)[0]==10:
			print('Gave an array of afew values')
			if not np.shape(waves)[0]==np.shape(afew)[0]:
				assert (np.shape(waves)[0]==np.shape(afew)[0]), "Array of afew values must be the same length as the number of wavelength bins, which is"+str(np.shape(waves)[0])
			nparams=int(afew[counter]+2)
		else:
			if not isinstance(afew,int):
				assert isinstance(afew,int), "afew must be an integer >=1!"
			elif afew>=16:
				print('Performing fit for best number of eigencurves to use.')
				delbic=20.
				nparams=4
				params0=10.**-4.*np.ones(nparams)
				mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams))
				resid=mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams)
				chi2i=np.sum((resid//eclipseerrors)**2.)
				loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
				bici=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
			#print(nparams,chi2i,bici,mpfit[0])
				tempparams=mpfit[0]
			#pdb.set_trace()
				while delbic>10.:#sf>0.00001:
					nparams+=1
					if nparams==15:
						params0=10.**-4.*np.ones(nparams)
						mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams))
						chi2f=np.sum((mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams)//eclipseerrors)**2.)
						#dof=np.shape(eclipseerrors)[0]-nparams
						#Fval=(chi2i-chi2f)/(chi2f/dof)
						#sf=0.00000001
						delbic=5.
					else:
						params0=10.**-4.*np.ones(nparams)
						mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams))
						resid=mpmodel(mpfit[0],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams)
						chi2f=np.sum((resid//eclipseerrors)**2.)
						#dof=np.shape(eclipseerrors)[0]-nparams
						#Fval=(chi2i-chi2f)/(chi2f/dof)
						#sf=stats.f.sf(Fval,nparams-1,nparams)
						loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
						bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])
						delbic=bici-bicf
						#pdb.set_trace()
						#print(np.sum((resid//eclipseerrors)**2),loglike)
						#print(chi2i,chi2f,bici,bicf)
						print(nparams-2,bicf-bici)#,chi2f-chi2i,bicf-bici)
						print(mpfit[0])
						print(mpfit[0][:-1]-tempparams)
						chi2i=chi2f
						bici=bicf
						tempparams=mpfit[0]
				print('BIC criterion says the best number of eigencurves to use is '+str(nparams-3))

			#pdb.set_trace()
				nparams-=1	#need this line back when I change back again
			
			elif ((afew<16)&(afew>=1)):
				nparams=int(afew+2)

			else:	#assert afew is an integer here
				assert afew>1 ,"afew must be an integer 1<=afew<=15!"
		#nparams=5
		params0=10.**-4.*np.ones(nparams)
		
		#pdb.set_trace()
		mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams))
		#format parameters for mcmc fit
		theta=mpfit[0]
		ndim=np.shape(theta)[0]	#set number of dimensions
		#print(mpfit[0])
		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,escore,nparams),threads=6)
		pos = [theta + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]
		
		print("Running MCMC at {} um".format(waves[counter]))

		bestfit=np.zeros(ndim+1)
		bestfit[0]=10.**8

		#sampler.run_mcmc(pos,nsteps)
		for i, result in enumerate(sampler.sample(pos, iterations=nsteps)):
			if i>burnin:
				for guy in np.arange(nwalkers):
					resid=mpmodel(result[0][guy],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams)
					chi2val=np.sum((resid//eclipseerrors)**2.)
					if chi2val<bestfit[0]:
						bestfit[0]=chi2val
						bestfit[1:]=result[0][guy]

		resid=mpmodel(bestfit[1:],eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore),nparams)
		chi2f=np.sum((resid//eclipseerrors)**2.)
		loglike=-0.5*(np.sum((resid//eclipseerrors)**2 + np.log(2.0*np.pi*(eclipseerrors)**2)))
		bicf=-2.*loglike + nparams*np.log(np.shape(eclipseerrors)[0])

		samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
		#pdb.set_trace()
		def quantile(x, q):
			return np.percentile(x, [100. * qi for qi in q])

		# bestcoeffs=np.zeros(np.shape(samples)[1])
		# for i in np.arange(np.shape(bestcoeffs)[0]):
		# 	bestcoeffs[i]=quantile(samples[:,i],[0.5])
		bestcoeffs=bestfit[1:]

		# plt.figure()
		# plt.plot(eclipsetimes,bestcoeffs[2]*escore[0,:])
		# plt.plot(eclipsetimes,bestcoeffs[3]*escore[1,:])
		# plt.plot(eclipsetimes,bestcoeffs[4]*escore[2,:])
		# plt.show()
		#print(bestcoeffs)
		#pdb.set_trace()
		# translate coefficients
		fcoeffbest=np.zeros_like(ecoeff)
		for i in np.arange(np.shape(bestcoeffs)[0]-2):
			fcoeffbest[:,i] = bestcoeffs[i+2]*ecoeff[:,i]

		# how to go from coefficients to best fit map
		spheresbest=np.zeros(int((degree)**2.))
		for j in range(0,len(fcoeffbest)):
			for i in range(1,int((degree)**2.)):
				spheresbest[i] += fcoeffbest.T[j,2*i-1]-fcoeffbest.T[j,2*(i-1)]
		spheresbest[0] = bestcoeffs[0]#c0_best
		bestfitoutput[:,counter]=spheresbest
		#pdb.set_trace()
		for sampnum in np.arange(np.shape(samples)[0]):
			fcoeff=np.zeros_like(ecoeff)
			for i in np.arange(np.shape(samples)[1]-2):
				fcoeff[:,i] = samples[sampnum,i+2]*ecoeff[:,i]

			# how to go from coefficients to best fit map
			spheres=np.zeros(int((degree)**2.))
			for j in range(0,len(fcoeff)):
				for i in range(1,int((degree)**2.)):
					spheres[i] += fcoeff.T[j,2*i-1]-fcoeff.T[j,2*(i-1)]
			spheres[0] = samples[sampnum,0]#bestcoeffs[0]#c0_best	
			
			alltheoutput[sampnum,:,counter]=spheres
		#pdb.set_trace()
		#Translate all the coefficients for all of the posterior samples
		#alltheoutput[counter,1:]=spheres
		#print(spheresbest)

		if plot:
			params0=sp.ModelParams(brightness_model='spherical')	#no offset model
			params0.nlayers=20

			params0.t0=-2.21857/2.				# Central time of PRIMARY transit [days]
			params0.per=2.21857567			# Period [days]
			params0.a_abs=0.0313			# The absolute value of the semi-major axis [AU]
			params0.inc=85.71			# Inclination [degrees]
			params0.ecc=0.0			# Eccentricity
			params0.w=90.			# Argument of periastron
			params0.rp=0.155313				# Planet to star radius ratio
			params0.a=8.863				# Semi-major axis scaled by stellar radius
			params0.p_u1=0.			# Planetary limb darkening parameter
			params0.p_u2=0.			# Planetary limb darkening parameter

			params0.degree=degree	#maximum harmonic degree
			params0.la0=0.
			params0.lo0=0.
			params0.sph=list(spheresbest)

			times=eclipsetimes
			templc=params0.lightcurve(times)

			params0.plot_square()

			# doing a test spiderman run to see if the output lightcurve is similar
			# #PERSON CHECKING THIS: You can use this to make sure the spherical harmonics fit is doing the right thing!

			plt.figure()
			plt.plot(times,templc,color='k')
			plt.errorbar(eclipsetimes,eclipsefluxes,yerr=eclipseerrors,linestyle='none',color='r')
			plt.show()

		nParamsUsed.append(nparams)
		ecoeffList.append(ecoeff)
		escoreList.append(escore)
		#elcList.append(elc)
		eigencurvecoeffList.append(samples)
		elatentList.append(elatent)
		
	
	finaldict={'wavelength (um)':waves,'spherical coefficients':alltheoutput,'best fit coefficients':bestfitoutput,'N Params Used':nParamsUsed,
				'ecoeffList': ecoeffList,'escoreList': escoreList,'elc': elc,'eigencurve coefficients':eigencurvecoeffList,'BIC':bicf,'elatentList':elatentList}
	return finaldict
