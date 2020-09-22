import spiderman as sp
import numpy as np

def spider_model(n_layers=20,t0=0,per=2.21857567,a_abs=0.0313,inc=85.71,ecc=0.0,w=90,rp=0.155313,a=8.863,p_u1=0,p_u2=0,ntimes=500,coeff=1,sph=0,degree=3):
    """ Create initial spiderman model """
    # spherical harmonics and 20 layers (WILL NEED TO TEST # OF LAYERS EVENTUALLY)
    spider_params = sp.ModelParams(brightness_model='spherical')
    spider_params.n_layers= n_layers  # (default)

    # SHOULD ALSO TEST DATA AGAINST UNIFORM BRIGHTNESS MODEL (A LA DEWIT) TO GET SIGMA-DETECTION OF STRUCTURE

    # for the system parameters we'll use HD 189733b (from Agol et al. 2010, except where noted)
    spider_params.t0= t0            # Central time of PRIMARY transit [days]
    spider_params.per= per          # Period [days]
    spider_params.a_abs= a_abs      # The absolute value of the semi-major axis [AU] -- from Southworth 2010
    spider_params.inc= inc          # Inclination [degrees]
    spider_params.ecc= ecc          # Eccentricity
    spider_params.w= w              # Argument of periastron -- arbitrary, MAY NEED TO CHANGE IF E NE 0
    spider_params.rp= rp            # Planet to star radius ratio
    spider_params.a= a              # Semi-major axis scaled by stellar radius
    spider_params.p_u1= p_u1        # Planetary limb darkening parameter
    spider_params.p_u2= p_u2        # Planetary limb darkening parameter
    spider_params.degree= degree            # Maximum degree of harmonic (-1): 3 means 0th +8 components (x2 for negatives)
    spider_params.la0= 0                    # Offset of co-ordinte centre from the substellar point in latitude (Degrees)
    spider_params.lo0= 0                    # Offset of co-ordinte centre from the substellar point in longitude (Degrees)
    
    
    return spider_params
    

def sh_lcs(n_layers=20,t0=0,per=2.21857567,a_abs=0.0313,inc=85.71,ecc=0.0,w=90,rp=0.155313,a=8.863,p_u1=0,p_u2=0,ntimes=500,coeff=1,sph=0,degree=3):

    spider_params = spider_model(n_layers=n_layers,t0=t0,per=per,a_abs=a_abs,inc=inc,ecc=ecc,w=w,rp=rp,
                                 a=a,p_u1=p_u1,p_u2=p_u2,ntimes=ntimes,coeff=coeff,sph=sph,degree=degree)
    # brightness model parameters
    # and we'll want to calculate lightcurves for each individual component, so that they can be fed into PCA
    # in Veenu's work we used lmax=4 (=49 components, incl. negative SHs)
    #   our best fit was 0th + 4 eigen-components
    #   but our information content didn't drop to noise until after the 25th component (implies degree=4)
    if degree<=6:
        if np.size(ntimes) == 1:
            t= spider_params.t0 + np.linspace(0, spider_params.per,ntimes)  # TEST TIME RESOLUTION
        else:
            t= ntimes
            ntimes = t.size
        
        if np.size(sph) == 1:
        
            allLterms = [0] * degree**2
            allLterms[0] = 1
            spider_params.sph= allLterms * coeff   # this is l0, so don't need a negative version
            lc = spider_params.lightcurve(t)
            # set up size of lc to be able to append full set of LCs
            lc = np.resize(lc,(1,ntimes))
        
            spider_params.sph= [0] * degree**2
            # set up 2-d array of LCs for all SHs
            for i in range(1,len(spider_params.sph)):
                spider_params.sph[i]= -1*coeff
                tlc = spider_params.lightcurve(t)
                tlc = np.resize(tlc,(1,ntimes))
                lc = np.append(lc,tlc,axis=0)
                spider_params.sph[i]= 1*coeff
                tlc = spider_params.lightcurve(t)
                tlc = np.resize(tlc,(1,ntimes))
                lc = np.append(lc,tlc,axis=0)
                spider_params.sph[i]= 0    
        else:        
        # calcualte single lightcurve for single set of spherical harmonic coefficients
            spider_params.sph= sph   # spherical harmonic coefficients
            lc = spider_params.lightcurve(t)
    
    else:
        assert (degree>6),"Can't handle this high of a spherical harmonic degree!"
    
    #sestart= np.int(ntimes*.475)
    #seend= np.int(ntimes*.525)
    #et = t[sestart:seend]
    #elc = lc[:,sestart:seend]
    #ot = np.append(t[0:sestart],t[seend:])
    #olc = np.append(lc[:,0:sestart],lc[:,seend:],axis=1)

    #return lc,elc,olc,t,et,ot

    return lc,t
