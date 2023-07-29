from scipy.special import wofz
import numpy as np
import os
from astropy.convolution import convolve, Gaussian1DKernel
import emcee
from numpy.random import randn
import matplotlib as mpl
import copy

if (False):
    mpl.use('qt5Agg')
    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from 
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 12,               # LaTeX default is 10pt font.
        "font.size": 10,
        "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
    mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt

#FWHM=4 pixels (50 km/s, 12.5 km/s/pixel, 2.355 converts to sigma)
fire_kernel = Gaussian1DKernel(stddev=4/2.355) 

#6km/s, 1.4 km/s/pixel (careful, i have different verision with different binning)
hires_kernel= Gaussian1DKernel(stddev=4.285/2.355) 

xshooter_kernel= Gaussian1DKernel(stddev=2.9/2.355) 

####DO NOT TOUCH THESE! SHOULD PROBABLY PUT IN INIT FILE
b_const=0.0166289 #b constant=2*k/(amu/gram), in (km/s)^2/K, to calculate b from b_T and T
c=29979245800.0 #cm/s
voigt_const=448898479.507 #sqrt(pi)*e^2/m_e in cm^3/s^2, voigt profile constant, needs to be divided by rest_freq*b
####

######################### VOIGT PROFILES ###############
#
# Compute Voigt profile optical depth.
# This is N*f*const*H(a,x)
# const=sqrt(pi)*e^2/(m_e*freq_rest*b)
# H(a,x) is real part of Faddeeva function
# a=gamma*c/(4*pi*freq_rest*b)
# x=(freq-freq0)*lambda_0/b
# (See Fundamentals of Stellar Astrophysics by Collins)
# There are faster approximations to the Voigt profile that can be used

# lambda0: rest wavelenght in Angstrom
# b:     Doppler parameter in km/s
# f:     oscillator strength
# N:     Log of column density in cm^-2 (Used to not be log)
# gamma: gamma of transition in 1/s
# wv:    wavelength grid on which to calculate the profile
def voigt(lambda0,gamma,f,N,b,z,wv):
    wv_r=wv/(1+z) #Correct wavelengths to be fit to rest frame
    #c=29979245800.0 #cm/s
    b_f=b/lambda0*10**13 #Doppler frequency, constant accounts for A,km conversion. Units of Hz
    a=gamma/(4*np.pi*b_f) #Dimensionless damping parameter of intrinsic line shape
    freq0=c/lambda0*10**8
    constant=voigt_const/(freq0*b*10**5) #10^5 is b from km/s->cm/s
    freq=c/wv_r*10**8 #10^8 is cm/s->A/s conversion
    x=(freq-freq0)/b_f #Dimensionless input to H(a,x)
    H=np.real(wofz(x+1j*a))
    tau=(10**N)*f*constant*H
    return tau

def vpFromModel(model,spec):
    wv=spec.specw
    tau = 0 * wv
    for component in model.components.values():
        Nsub=component.Nsub
        dvsub=component.dvsub
        z=component.z
        b_turb=component.b_turb
        for ion in component.ions.values():
            N = ion.N
            if (Nsub > 1):
                Ntmp = N - np.log10(Nsub)
            for lkey,line in ion.transitions.items():
                if lkey not in spec.lines: continue
                #This is called by iterating over spectra, and fitregions take into account which spec is used--should be fine!
                restwv = line.wave0
                gamma  = line.gamma
                f      = line.f
                if (Nsub == 1):
                    # Default: one subcomponent in a component (i.e. "resolved")
                    if component.T==None:
                        b=b_turb
                    else:
                        b=np.sqrt(b_const*(10**component.T)/line.mass+b_turb**2)
                    tau+=voigt(restwv,gamma,f,N,b,z,wv)
                else:
                    for icomp in range(Nsub):
#                         # Divide total column into Nsub equal subcomponents
#                         # spread in velocity space.  Fitting total N only,
#                         # This is mostly for testing purposes.
                        vdelt = dvsub / Nsub
                        zdelt = vdelt / 300000. * (1+z)
                        ztmp = z + (Nsub/2-icomp) * zdelt
                        if component.T==None:
                            b=b_turb
                        else:
                            b=np.sqrt(b_const*(10**component.T)/line.mass+b_turb**2)
                        tau+=voigt(restwv,gamma,f,Ntmp,b,ztmp,wv)
    return tau


#hmm--want to do a quick vpfit..probably should use mpfit
#Currently non-functional
def maxlike(model):
    #Make this work!
    pass #:(
    # import scipy.optimize as op
    # # nll = lambda *args: -lnlike(*args)
    # neglike = lambda th,model: -likeli(th,model)
    # result = op.minimize(neglike, model2List(model), args=(model))
    # return result

#Convert optical depth to flux
#kernel is Gaussian Kernel, should be included in spectra class
def vpTau2Flux(tau, kernel):
    fmodel=convolve(np.exp(-tau),kernel,boundary='extend')
    return fmodel

##################### END VOIGT PROFILE CALC #################
#deprecated?
def generateFittingMask(model):
    wv = model.spectra['FIRE'].specw
    mask = np.repeat(False,len(wv))
    for component in model.components.values():
        z=component.z
        for ion in component.ions.values():
            for line in ion.transitions.values():
                restwv = line.wave0
                for reg in line.fitregions:
                    w0   = (1+z) * restwv
                    wmin = w0 * (1 + reg.vmin/(c/1e5))
                    wmax = w0 * (1 + reg.vmax/(c/1e5))
                    mask[np.logical_and((wv>wmin),(wv<wmax))] = True
                    #cwl    = (1+z)*restwv
                    #vspec  = np.array((wv/cwl-1)*(c/1e5))
                    #mask[np.logical_and((vspec > vmin),(vspec < vmax))] = True
    return mask

#TC:Haven't touched this at all
def equivalentWidths(model,wv,spec,err):

    dlam = wv - np.roll(wv,1)
    dlam[0] = dlam[1]

    comp_redshifts = model.components.keys()
    for z in comp_redshifts:
        print("z = {}".format(z))
        ions=model.components[z].ions.keys()
        for ion in ions:
            transitions=model.components[z].ions[ion].transitions.keys()
            for line in transitions:
                mask = np.repeat(False,len(wv))
                restwv = model.components[z].ions[ion].transitions[line].wave0
                zl      = model.components[z].z
                fitreg = model.components[z].ions[ion].transitions[line].fitregions.keys()
                for reg in fitreg:
                    vmin   = model.components[z].ions[ion].transitions[line].fitregions[reg].vmin
                    vmax   = model.components[z].ions[ion].transitions[line].fitregions[reg].vmax
                    cwl    = (1+zl)*restwv
                    vspec  = np.array((wv/cwl-1)*(c/1e5))
                    mask[np.logical_and((vspec > vmin),(vspec < vmax))] = True
                    ew    = np.sum((1.0-spec[mask])*dlam[mask])/(1+zl)
                    ewerr = np.sqrt(np.sum(err[mask]**2*dlam[mask]))/(1+zl)
                print("{},{}:{}+/-{}",format(ion,line,ew,ewerr))



##################### START LIKELIHOODS ######################

##################### DEFAULT LIKELIHOOD FUNCTION ###############

# #
# # Takes a model, and turns it into a list of variables to be fitted
# # (a THETA list for emcee)
# # Order heirarchy is component z, component b, component ion (N_)
# def model2List(m):
#     theta = []
#     for component in m.components.values():
#         if component.zpriors!=None: theta.append(component.z)
#         #hmm..what happens if we just give it b?
#         if component.bpriors!=None: theta.append(component.b_turb)
#         if component.T!=None and component.Tpriors!=None: theta.append(component.T)
#         for ion in component.ions.values():
#             if ion.Npriors!=None: theta.append(ion.N)
#     return theta

# #
# # This method translates an array of numbers, which is the format required
# # by the emcee walkers, into an instance of a voigt profile model class.
# # It needs an example model to map the nesting of redshift, N, and b
# #
# def list2Model(m,theta):
# #    tt = copy.deepcopy(m) #so that m isn't modified as well
#     tt = m.copymodel() #so that m isn't modified as well
#     ind=0
#     for component in tt.components.values():
#         if component.zpriors!=None: 
#             component.z = theta[ind] 
#             ind+=1
#         if component.bpriors!=None: 
#             component.b_turb = theta[ind] 
#             ind+=1
#         if component.T!=None and component.Tpriors!=None: 
#             component.T=theta[ind]
#             ind+=1
#         for ion in component.ions:
#             if ion.Npriors!=None: 
#                 ion.N=theta[ind]
#                 ind+=1
#     return tt

####### Check prior bounds #######

def meetsPriors(m):
    # Minimum and maximum priors, contained in the 
    # model class.
    for component in m.components.values():
        z = component.z
        if component.bpriors!=None: 
            b_turb = component.b_turb
            bmin,bmax=component.bpriors
            if(b_turb < bmin or b_turb > bmax): 
                return False
        if component.zpriors!=None:
            zmin,zmax=component.zpriors
            if(z < zmin or z > zmax): 
                return False
        T=component.T
        if T!=None and component.Tpriors!=None:
            Tmin,Tmax=component.Tpriors
            if(T<Tmin or T>Tmax):
                return False
        for ion in component.ions.values():
            if ion.Npriors!=None:
                N=ion.N
                Nmin,Nmax=ion.Npriors
                if (N < Nmin or N > Nmax): 
                    return False
    return True

def generateWalkerPosition(m):
    #theta = np.array([])
    # pos   = np.array([])
    pos=[]

    for component in m.components.values():
        if component.zpriors!=None:
            pos.append(component.z+0.0001*randn(1))
        if component.bpriors!=None:
            prop = component.b_turb+2*randn(1)
            if (prop > 0.5):
                pos.append(prop)
            else:
                pos.append(np.array([0.5]))
        if component.T!=None and component.Tpriors!=None:
            pos.append(component.T+0.5*randn(1))
        #theta = np.append(theta,m.components[z].z+0.00001*randn(1))
        # b = m.components[z].b+0.1*randn(1)
        # theta = np.append(theta,b)
        # ionkeys=m.components[z].ions.keys()
        for ion in component.ions.values():
            if ion.Npriors!=None:
                pos.append(ion.N+0.2*randn(1))

    #pos is a list of numpy arrays--make it a 1d array
    pos=np.concatenate(pos)
    return pos

#Theta is in the order as constucted by model2list
#TC: This can be sped up by for a given ion only fitting it's fitregion. I.e, we are making a lot of 0 optical depth calculations
def likeli(theta,m):
    # Emcee gives us the theta vector, repopulate this into
    # a temporary model buffer for calculating the voigt profile
    model_temp=m.list2Model(theta) #no copy is made, so really m and model_temp point ot the same thing
   
    #If passes priors test
    if meetsPriors(model_temp):
        # Generate the voigt profile
        prob=0
        #iterate over the spectra
        for name,spec in m.spectra.items():
            tau=vpFromModel(model_temp,spec)
            model_flux=vpTau2Flux(tau,spec.kernel)
            model_flux=model_flux[spec.w2fit] #trim extra bits added on for convolution
            # Calculate the likelihood (~chi^2)
            # inv_sigma2 = 1.0/(spece[mask]**2)
            inv_sigma2 = 1.0/(spec.spece**2)
            prob+=-0.5*(np.sum((spec.specf-model_flux)**2*inv_sigma2+np.log(inv_sigma2)))
#            prob+=-0.5*(np.sum((spec.specf-model_flux)**2*inv_sigma2))
 #           tt = -0.5*(np.sum((spec.specf-model_flux)**2*inv_sigma2+np.log(inv_sigma2)))
#            print("Increment: {}".format(tt))
#        print("Inside priors")
        return prob
    else: # Outside of prior boundaries, prob is -inf
        return -np.inf

######################## END DEFAULT LIKELIHOOD FUNCTION #####################

#########################  Core MCMC model fitter ###############

# m is the model that is input, used for setup and passed to voigt profile generator
# For now wfit is the positions in the wavelength array where I want to fit things, 
# NOT the wavelengths

# def runmc(m,specw,specf,spece,nwalkers=50,nruns=2000):
def runmc(m,nwalkers=60,nruns=2000):
    # theta = model2List(m)
    # ndim = len(theta)
    pos  = [generateWalkerPosition(m) for i in range(nwalkers)]
    ndim = np.shape(pos)[1]

    model=m.copymodel()
    model.cropSpec()

    sampler=emcee.EnsembleSampler(nwalkers,ndim,likeli,args=[model],moves=emcee.moves.StretchMove(a=2))
    sampler.run_mcmc(pos,nruns,progress=True)
    return sampler


#########################################################################

def vplot(specobj, model, vp=None, fname='velo_plot.pdf',linelist=None):
    
    waves = specobj.specw
    flux  = specobj.specf
    err   = specobj.spece

    zind = list(model.components.keys())
#    if (len(zind) == 1):
    zavg = model.components[zind[0]].z
#    else:
#        zavg = np.mean(model.components[zind].z)

    plotwaves  = []
    potentials = []
    ionnames   = []

#    print linelist

    for iz in zind:
        ionkeys=model.components[iz].ions.keys()
        for iion in ionkeys:
            transkeys=model.components[iz].ions[iion].transitions.keys()
            for it in transkeys:
                restwv = model.components[iz].ions[iion].transitions[it].wave0
                if (restwv in plotwaves): continue
                if (not(np.floor(restwv) in linelist)): 
#                    print restwv
                    continue
                E_ion = model.components[iz].ions[iion].transitions[it].E_ion
                plotwaves  = np.append(plotwaves,restwv)
                ionnames.append(iion)
                potentials = np.append(potentials,E_ion)

    # This little slice of magic sorts plotwaves in order of increasing potential.
    # We will plot from bottom up, in order of increasing ionization potential
    ind = np.argsort(potentials)
    ionsort = []
    for ii in range(len(ind)):
        ionsort.append(ionnames[ind[ii]])
    plotwaves=([plotwaves for potentials,plotwaves in sorted(zip(potentials,plotwaves))])
    nplot=len(plotwaves)

    dv=1000 # Default velocity width, km/s

    fig = plt.figure(figsize=(5/0.875,1.618*5))
#    fig = plt.figure(figsize=(5*1.0,1.0*5))
    ax  = fig.add_subplot(111)

    ax.set_xlim(-1.0*dv,1.0*dv)
    ax.set_ylim(-1,2*nplot)
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Normalized Flux")

    for i in range(nplot):
        cwl = (1+zavg)*plotwaves[i]
        print("{}".format(str(ionsort[i])))
        v   = (waves/cwl-1)*(c/1e5)

        plotflux = np.copy(flux)
        plotflux[np.logical_and((flux < 2*err),(np.abs(v) > 150.),(flux > 0.3))] = None
        plotflux[flux > 1.5] = None
        ax.plot(v,plotflux+2*i,drawstyle='steps-mid',linewidth=2,color="k")

        # This plots "bad" points using a transparent color.
        plotflux = np.copy(flux)
        ax.plot(v,plotflux+2*i,drawstyle='steps-mid',linewidth=2,color="k",alpha=0.05)

        ax.plot(v,err+2*i,drawstyle='steps-mid',color="b",alpha=0.2)
        label = ionsort[i]+"$\lambda$"+str(plotwaves[i]).split('.')[0]
        ionlab=ax.annotate(label,xy=(dv*1.05,2*i+0.75),xycoords='data',annotation_clip=False)
        #Voigt profiles are given, include in vplot
        if vp!=None:
            nprof = len(vp)
            for iprof in range(nprof):
                ax.plot(v,vp[iprof]+2*i,color="r",alpha=0.10)

    ax.set_yticks(np.arange(-1,2*nplot,1)) # Set to plot grid lines at every y integer
    #ax.axes.get_yaxis().set_ticklabels([]) # Suppress label numbers on y
    plt.grid()
    plt.tight_layout(rect=(0.01,0.01,0.875,0.97))

    fig.savefig(fname)

#    plt.show()
    
########################
#Haven't tested with new changes

def sampleVPFits (model, specobj, samples, nsamp):
    draws = samples[np.random.randint(len(samples),size=nsamp)]
#    tm = model.list2Model(model,draws[3])
    profs = [vpTau2Flux(vpFromModel(model.list2Model(draws[i]),specobj),specobj.kernel) for i in range(len(draws))]
    return profs
