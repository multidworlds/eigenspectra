#Function to fit a map at each wavelength using Eigencurves

#THINGS THAT WILL BE UPDATED IN FUTURE VERSIONS:
#	1. Right now it just loads in a single file, and for the entire pipeline it will load in a file for each wavelength and run the fit for each wavelength
#	2. It will eventually output wavelengths in addition to the spherical harmonic coefficients
#	3. Right now it just selects a number of eigencurves to include. Will eventually optimize this from chi-squared.

#INPUTS:
#	Secondary eclipse light curves at each wavelength (csv for each wavelength)

#OUTPUTS:
#	Coefficients for each of the spherical harmonics
from lightcurves_sh import sh_lcs
from pca_eig import princomp
import numpy as np 
import matplotlib.pyplot as plt 
import emcee
import csv
import spiderman as sp
#from importlib import import_module
#import glob
#import os
from scipy.optimize import leastsq
import pdb

def mpmodel(p,x,y,z,elc,escore):#fjac=None,x=None,y=None,err=None):
	model = p[0]*elc[0,:] + p[1]
	for ind in range(2,len(p)):
		model = model + p[ind] * escore[ind-2,:]
	return np.array(y-model)

def lnprob(theta,x,y,yerr,elc,escore):
	lp=lnprior(theta)
	return lp+lnlike(theta,x,y,yerr,elc,escore)

def lnlike(theta,x,y,yerr,elc,escore):
	c0,fstar,c1,c2,c3,c4,c5,c6=theta
	model = c0*elc[0,:] + fstar + c1*escore[0,:] + c2*escore[1,:] + c3*escore[2,:] + c4*escore[3,:] + c5*escore[4,:] + c6*escore[5,:]
	resid=y-model
	chi2=np.sum((resid/yerr)**2)
	dof=np.shape(y)[0]-1.
	chi2red=chi2/dof
	ln_likelihood=-0.5*(np.sum((resid/yerr)**2 + np.log(2.0*np.pi*(yerr)**2)))
	return ln_likelihood

def lnprior(theta):
	lnpriorprob=0.
	c0,fstar,c1,c2,c3,c4,c5,c6=theta
	if fstar<0.:
		lnpriorprob=-np.inf
	elif c0<0.:
		lnpriorprob=-np.inf
	return lnpriorprob

def eigencurves(dict,plot=False,sph_harm_degree=3):

	#flist=glob.glob(os.path.join(directory,'*.csv'))
	waves=dict['wavelength (um)']
	times=dict['time (days)']
	fluxes=dict['flux (ppm)']	#2D array times, waves
	errors=dict['flux err (ppm)']
	alltheoutput=-np.ones((np.shape(waves)[0],1+sph_harm_degree**2))	#This file is going to have wavelength in the first column, and then the spherical harmonics after that

	if np.shape(fluxes)[0]==np.shape(waves)[0]:
		rows=True
	elif np.shape(fluxes)[0]==np.shape(times)[0]:
		rows=False
	else:
		assert (np.shape(fluxes)[0]==np.shape(times)[0]) | (np.shape(fluxes)[0]==np.shape(waves)[0]),"Flux array dimension must match wavelength and time arrays."

	#counter=0
	#for f in flist:
	for counter in np.arange(np.shape(waves)[0]):
		#Import data from csv files
		#myfile=np.loadtxt(f,delimiter=',',skiprows=1)
		wavelength=waves[counter] #wavelength this secondary eclipse is for
		eclipsetimes=times	#in days
		if rows:
			eclipsefluxes=fluxes[counter,:]*10.**-6.
			eclipseerrors=errors[counter,:]*10.**-6.
		else:
			eclipsefluxes=fluxes[:,counter]*10.**-6.
			eclipseerrors=errors[:,counter]*10.**-6.

		alltheoutput[counter,0]=wavelength 	#I need to determine how to read wavelengths from the csv file names

		#	Calculate spherical harmonic maps using SPIDERMAN
		# create a set of SH light curves
		#planet_name='HD189733b'
		#system=import_module('data.planet.{}'.format(planet_name))

		#lc,t = sh_lcs(t0=-2.21857/2.)	#times in days, and fp/fs - model at higher time resolution (default is ntimes=500)
		lc,t = sh_lcs(t0=-2.21857/2.,ntimes=eclipsetimes,degree=sph_harm_degree)	#model it for the times of the observations
		#lc,t = sh_lcs(system.properties,t0=-2.21857//2.,ntimes=eclipsetimes)

		#Optional add-on above: test whether we want to include higher-order spherical harmonics when making our eigencurves?

		# subtract off stellar flux
		lc = lc-1

		# just analyze time around secondary eclipse (from start to end of observations)
		starttime=np.min(eclipsetimes)
		endtime=np.max(eclipsetimes)
		ok=np.where((t>=starttime) & (t<=endtime))
		et = t[ok]
		elc=np.zeros((np.shape(lc)[0],np.shape(ok)[1]))
		for i in np.arange(np.shape(lc)[0]):
			elc[i,:] = lc[i,ok]

		# plt.figure()
		# plt.plot(et,elc[0,:],'bo')
		# plt.show()

		# plt.figure()
		# plt.plot(et,elc[1,:],'bo')
		# plt.show()

		#  PCA
		ecoeff,escore,elatent = princomp(elc[1:,:].T)

		#Optional add-on: test whether we want to fit for more than the first 4 eigencurves?
		#In here, make a plot showing the coefficients on the eigencurves from the best fit. Then we can set some sensitivity level and say "select the number of eigencurves that have above this level"

		#FIGURE OUT SOME STATISTICS
		#do an initial least squares fit?
		escore=np.real(escore)

		params0=np.ones(2 * sph_harm_degree**2)
		
		mpfit=leastsq(mpmodel,params0,args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,np.array(escore)))

		#format parameters for mcmc fit
		theta=mpfit[0]
		ndim=np.shape(theta)[0]	#set number of dimensions
		nwalkers=16 #number of walkers	

		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(eclipsetimes,eclipsefluxes,eclipseerrors,elc,escore),threads=6)
		pos = [theta + 1e-5*np.random.randn(ndim) for i in range(nwalkers)]

		burnin=200
		nsteps=1000

		sampler.run_mcmc(pos,nsteps)

		samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
		#fig = corner.corner(samples)
		#fig.savefig('temp.png')

		def quantile(x, q):
			return np.percentile(x, [100. * qi for qi in q])

		bestcoeffs=np.zeros(np.shape(samples)[1])
		for i in np.arange(np.shape(bestcoeffs)[0]):
			bestcoeffs[i]=quantile(samples[:,i],[0.5])

		# translate coefficients
		fcoeff=np.zeros_like(ecoeff)
		for i in np.arange(np.shape(bestcoeffs)[0]-2):
			fcoeff[:,i] = bestcoeffs[i+2]*ecoeff[:,i]

		# how to go from coefficients to best fit map
		spheres=np.zeros(sph_harm_degree**2)
		for j in range(0,len(fcoeff)):
			for i in range(1,sph_harm_degree**2):
				spheres[i] += fcoeff.T[j,2*i-1]-fcoeff.T[j,2*(i-1)]
			spheres[0] = bestcoeffs[0]#c0_best
		
		alltheoutput[counter,1:]=spheres
		#counter+=1
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

		params0.degree=3	#maximum harmonic degree
		params0.la0=0.
		params0.lo0=0.
		params0.sph=list(spheres)

		times=eclipsetimes
		templc=params0.lightcurve(times)

		# doing a test spiderman run to see if the output lightcurve is similar
		# #PERSON CHECKING THIS: You can use this to make sure the spherical harmonics fit is doing the right thing!

		plt.figure()
		plt.plot(times,templc,color='k')
		plt.errorbar(eclipsetimes,eclipsefluxes,yerr=eclipseerrors,linestyle='none',color='r')
		plt.show()

	return alltheoutput
