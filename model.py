import numpy as np
import os
import math
import matplotlib.pyplot as plt
import vfit as vf
from astropy.convolution import convolve, Gaussian1DKernel
import copy
from collections import OrderedDict

#1 spectrum fits
#2 implement ratio fixing

#FWHM=4 pixels (50 km/s, 12.5 km/s/pixel, 2.355 converts to sigma)
fire_kernel = Gaussian1DKernel(stddev=4/2.355) 
#6km/s, 1.4 km/s/pixel 
hires_kernel= Gaussian1DKernel(stddev=4.285/2.355)
xshooter_kernel= Gaussian1DKernel(stddev=2.9/2.355) 

c=29979245800.0 #cm/s
vfitdir=os.getenv("MCVP")
# atomf=os.atomdat=os.path.join(os.path.dirname(__file__),'atomic_data.txt')
atomf=os.atomdat=os.path.join(vfitdir,'atomic_data.txt')
atomdat=np.genfromtxt(atomf,dtype=None,names=True,encoding=None)

############################
# These three classes specify how we organize absorption systems.  They are
# modeled as:
#
# component.ion.transition
#
# Because there is only one column density associated with an ion, but it can 
# have multiple transitions.  Likewise a single component can have multiple ions (e.g.
# SiII/SiIV) so this is useful for fitting line ratios and other tied parameters.
# Details below.  When constructing the likelihood function, this will help us
# keep control of how we loop over the MCMC variables.

# The COMPONENT class corresponds to all IONS at a particular redshift
# It is characterized only by a redshift, but can have different N 
# for different IONS. It has a single value of, but can differ for various
# ions because of thermal+turbulent relative contributions.  
# For unresolved systems, NSub is a way to test
# convergence of TOTAL column density for multiple blended/saturated components.
# set priors to None to specify that something is fixed

class Component:
    def __init__(self,z,b_turb=6.0,bpriors=[0,50],T=None,Tpriors=[0,7],Nsub=1,dvsub=0.0,zpriors=None):
        self.z = z
        self.Nsub = Nsub
        self.dvsub = dvsub
        self.ions=OrderedDict()
        self.bpriors=bpriors
        self.b_turb = b_turb
        self.T = T
        self.Tpriors=Tpriors

        if (zpriors == None):
            self.zpriors = [z-0.0015,z+0.0015]
        else:
            self.zpriors = zpriors
   #Return an exact duplicate of a component. Used by copymodel and copycomponent.
   #Made so that we don't need to duplicate this in copymodel and copycomponent. Note that whenever changes are made to any part of the model structure
   #this function MUST be updated. Hopefully all the copy issues were solely due to pandas.
    def copy(self):
        newc=Component(self.z,b_turb=self.b_turb,bpriors=self.bpriors,T=self.T,Tpriors=self.Tpriors,Nsub=self.Nsub, dvsub=self.dvsub)
        newc.zpriors=self.zpriors #need to treat zprior a bit special
        for ikey,ion in self.ions.items():
            newc.ions[ikey]=Ion(ion.name,ion.N,ion.Npriors)
            for tkey,tr in ion.transitions.items():
                newc.ions[ikey].transitions[tkey]=Transition(tr.wave0,tr.f,tr.gamma,tr.mass,tr.E_ion)
                #self.addtransition(np.floor(tr.wave0),ikey,key,tkey)
                for fr in tr.fitregions:
                    newc.ions[ikey].transitions[tkey].fitregions.append(FitRegion(fr.vmin,fr.vmax,fr.specname,fr.wave))
        return newc
#                    self.addfitregion(tkey,ikey,key,[fr.vmin,fr.vmax],fr.specname,fr.wave)
# The ION class corresponds to one ionization state, e.g. FeII, which has 
# one column density and one b but multiple TRANSITIONS of different 
# oscillator strength
class Ion:
    def __init__(self,name,N=14.0,Npriors=[11,16.5]):
        self.name=name
        self.N=N
        self.Npriors = Npriors
        self.transitions=OrderedDict()

# This corresponds to one particular line associated with an ION, 
# e.g. the 2344\AA line of FeII.  It is characterized by oscillator strength, 
# f, and gamma.
class Transition:
    def __init__(self,lambda0,f,gamma,mass,E_ion):    
        self.wave0=lambda0
        self.f=f
        self.gamma=gamma
        self.mass=mass #mass and E_ion should be at ion level?
        self.E_ion=E_ion
        self.fitregions=[] #just a normal list

    def bTot(self,b_turb,T):
        return np.sqrt(vf.b_const*(10**T)/self.mass+b_turb**2)
        # mask[np.logical_and((wv>reg.wavemin) & (wv<reg.wavemax))] = True
        #self.fitregions=[fitregion()]#pd.Series((fitregion()))

# Used to define the region around an absorption line for calculating the likelihood. 
# We associate the fitting regions with each ion, and any one ion can have multiple fitting
# regions (in the event that the user wants to selectively mask out subregions e.g. of noise 
# spikes around the transition in the spectrum.
#
######################### SPECTRUM CLASS #############

class Spectrum:
    def __init__(self,specw,specf,spece,kernel,lines=None):
        self.specw=specw
        self.specf=specf
        self.spece=spece
        self.kernel=kernel
        self.w2fit=None #When spectrum is cropped, specw is padded so conv is done correctly.
                        #w2fit is the elements of specw that will be fit
        if lines is None: lines=[]
        self.lines=lines

class FitRegion:
    def __init__(self,vmin=-15,vmax=15,specname=None,wave=False):
        self.vmin=vmin
        self.vmax=vmax
        self.specname=specname
        self.wave=wave

################ Model Class ########################

class Model:

    def __init__(self):
        self.components=OrderedDict()
        self.spectra=OrderedDict()

    #
    # Takes a model, and turns it into a list of variables to be fitted
    # (a THETA list for emcee)
    # Order heirarchy is component z, component b, component ion (N_)
    def model2List(self):
        theta = []
        for component in self.components.values():
            if component.zpriors is not None: theta.append(component.z)
            #hmm..what happens if we just give it b?
            if component.bpriors is not None: theta.append(component.b_turb)
            if component.T is not None and component.Tpriors is not None: theta.append(component.T)
            for ion in component.ions.values():
                if ion.Npriors is not None: theta.append(ion.N)
        return theta

    #
    # This method translates an array of numbers, which is the format required
    # by the emcee walkers, into an instance of a voigt profile model class.
    # It needs an example model to map the nesting of redshift, N, and b
    #Can probably do this with pop instead of an indexer
    def list2Model(self,theta,copy=False):
    #    tt = copy.deepcopy(m) #so that m isn't modified as well
        if copy:
            tt = self.copymodel() #so that m isn't modified as well
        else: 
            tt=self
        ind=0
        for component in tt.components.values():
            if component.zpriors is not None: 
                component.z = theta[ind] 
                ind+=1
            if component.bpriors is not None: 
                component.b_turb = theta[ind] 
                ind+=1
            if component.T is not None and component.Tpriors is not None: 
                component.T=theta[ind]
                ind+=1
            for ion in component.ions.values():
                if ion.Npriors is not None: 
                    ion.N=theta[ind]
                    ind+=1
        return tt            

    def addcomponent(self,z,b_turb=5.0,bpriors=[1,15],T=None,Tpriors=[0,7],Nsub=1,dvsub=0,zpriors=None,key=None):
        if key is None: key=z
        if key in self.components:
             raise KeyError("Err: a component with key "+key+" already exists")
        self.components[key]=Component(z,b_turb=b_turb,bpriors=bpriors,T=T,Tpriors=Tpriors,Nsub=Nsub, dvsub=dvsub, zpriors=zpriors)

    def addion(self,name,zkey,N=14.0,Npriors=[11,16.5],key=None):
        if key is None: key=name
        if key in self.components[zkey].ions:
            raise KeyError("Err: ion with key "+key+"in component "+zkey+"already exists")
        if (name in atomdat['ion']): 
            # print("Found in atomic database: {}".format(name))
            try:
                self.components[zkey].ions[key]=Ion(name,N,Npriors)
            except KeyError:
                print("Err: Requested redshift does not have a component defined")
        else: 
            print("Err: Requested ion is not in the atomic database ({})".format(name))

            ########################################

    def addtransition(self,restwv,ionkey,zkey,vminmax=None,specname=None,wave=False,key=None):
        try:
            if (restwv == 1216): restwv = 1215
            if (restwv == 2853): restwv = 2852
            w = np.where(np.array(list(map(math.floor,atomdat['wave']))) == math.floor(restwv))
            if (len(w[0]) == 0):
                print(f"Err: No ions in the database with requested rest wavelength {restwv}")
            else:
                if key is None: 
                    key=math.floor(restwv)
                    if (key == 1215): key=1216
                if key in self.components[zkey].ions[ionkey].transitions:
                    raise KeyError("Err: ion with key "+key+"in component "+zkey+"already exists")
                tablename=atomdat['ion'][w][0]
                if(tablename == self.components[zkey].ions[ionkey].name):
                    lambda0=atomdat['wave'][w][0]
                    gamma=atomdat['gamma'][w][0]
                    f=atomdat['f'][w][0]
                    mass=atomdat['mass'][w][0]
                    E=atomdat['E_ion'][w][0] #E is ionization energy in Ryd, not used in fitting but nice for organizing plots
                    # rounded=math.floor(restwv) # this is just to make indexing easier.
                    # if (rounded == 1215): rounded=1216
                    # if specname is None: specname=m.spectra.keys()[0]
                    redshift=self.components[zkey].z
                    self.components[zkey].ions[ionkey].transitions[key]=Transition(lambda0,f,gamma,mass,E)
                    if vminmax is not None: self.addfitregion(key,ionkey,zkey,vminmax,specname,wave=wave)
                else:
                    print("Err: Input rest wavelength does not match wavelengths in the database for this ion")
        except KeyError:
            if ((zkey in self.components.keys()) == 0):
                print("Err: Added transition at redshift where no component exists")
            else:
                print("Err: Added a transition for an ion that is not instantiated at this z")

                ###############################
    #wave=True means vminmax is actually lambda minmax
    def addfitregion(self,lkey,ikey,zkey,vminmax,specname,wave=False): 
        if isinstance(lkey,float):
            lkey=math.floor(lkey) # this is just to make indexing easier.
            if (lkey == 1215): lkey=1216
        try:
            fr = self.components[zkey].ions[ikey].transitions[lkey].fitregions
            nreg = len(fr)
            #indx = str((vminmax[0]+vminmax[1])/2.0)
            fr.append(FitRegion(vmin=vminmax[0],vmax=vminmax[1],specname=specname,wave=wave))
        except KeyError:
            if ((zkey in self.components.keys()) == 0):
                print("Err: Added transition at redshift where no component exists")
            else:
                print("Err: Added a transition for an ion that is not instantiated at this z")

                ##################################

    def addspec(self,name,specw,specf,spece,kernel,lines=None):
        self.spectra[name]=Spectrum(specw,specf,spece,kernel,lines)

    # This is a handrolled Deep Copy function for copying by value
    # from one model into a duplicate at a different memory location.
    # Useful while fitting so that the original copy doesn't get
    # overwritten.

          #######################################
    def copymodel(self):
        new = Model()
        for zkey,component in self.components.items():
            new.components[zkey]=component.copy()
            # new.addcomponent(component.z,b_turb=component.b_turb,bpriors=component.bpriors,T=component.T,
            #                  Tpriors=component.Tpriors,Nsub=component.Nsub,dvsub=component.dvsub,key=key)
            # for ikey,ion in component.ions.items():
            #     new.addion(ion.name,zkey,ion.N,ion.Npriors,key=key)
            #     for tkey,tr in ion.transitions.items():
            #         new.addtransition(np.floor(tr.wave0),ikey,zkey,tkey)
            #         for fr in tr.fitregions:
            #             new.addfitregion(tkey,ikey,zkey,[fr.vmin,fr.vmax],fr.specname,fr.wave)
        for name,spec in self.spectra.items():
            new.addspec(name,spec.specw,spec.specf,spec.spece,spec.kernel,spec.lines)

        return new

    #Make a new component that has the same structure, but different redshift so we don't need so many components
    #z1 is the already existing component, z2 is the one to copy it to
    #This is replaced by function below
    # def copycomponent(self,z1,z2):
    #     comp=copy.deepcopy(self.components[z1]) #deepcopy makes a copy of everything copied object points to--otherwise both components would be sharing ion objects
    #     comp.z=z2
    #     if self.components[z1].zpriors is not None: comp.zpriors=[z2-0.0001,z2+0.0001]
    #     self.components[z2]=comp

    # Deep Copy function that takes a component within a model
    # and copies it to another redshift *within the same model*.
    # This saves time when constructing initial guesses for 
    # later fitting, and gives new memory locations for everything.
    def copycomponent(self,z1key,z2,key=None):
        if key is None: key=z2
        try:
            old = self.components[z1key]
        except: KeyError('No component in model with at z1 key')
        newc=old.copy()
        newc.z=z2
        if newc.zpriors is not None:
            newc.zpriors=[x+z2-old.z for x in old.zpriors]
        self.components[key]=newc
        # self.addcomponent(z2,b_turb=old.b_turb,bpriors=old.bpriors,T=old.T,Tpriors=old.Tpriors,Nsub=old.Nsub,dvsub=old.dvsub,key=key)
        # for ikey,ion in old.ions.items():
        #     self.addion(ion.name,z2,ion.N,ion.Npriors,ikey)
        #     for tkey,tr in ion.transitions.items():
        #         self.addtransition(np.floor(tr.wave0),ikey,key,tkey)
        #         for fr in tr.fitregions:
        #             self.addfitregion(tkey,ikey,key,[fr.vmin,fr.vmax],fr.specname,fr.wave)

            ############################################

    #This is like the fitting mask--trim all the parts of the spectrum outside of fit regions
    #Problem--the convolution when making the VP is done incorrectly because random chunks 
    #are put next to each other
    #Solution is to pad the wavelength array so that it's longer
    #This should really figure out how much to pad based on the FWHM

    def cropSpec(self):
        for name,spec in self.spectra.items():
            specw=spec.specw
            fitmask=np.repeat(False,len(specw))
            wmask=np.repeat(False,len(specw)) #mask for wavelengths, longer than mask for fit
            todel=[]
            for component in self.components.values():
                for ion in component.ions.values():
                    for lkey,line in ion.transitions.items():
                        for reg in line.fitregions:
                            if reg.specname!=name: continue
                            if lkey not in spec.lines: spec.lines.append(lkey)
                            w0 = (1+component.z)*line.wave0
                            if reg.wave: #vminmax are wavelength
                                wfit=np.where((specw>reg.vmin)&(specw<reg.vmax))[0]
                                fitmask[wfit]=True
                                #padding by 10 pixels on either side, 10+1 so that when i change it ill remember how arange works
                                wwave=np.arange(min(wfit)-10,max(wfit)+10+1) 
                                wmask[wwave]=True
                            else:
                                specv = (specw-w0)/w0 * (c/1e5)
                                wfit=np.where((specv>reg.vmin)&(specv<reg.vmax))[0]
                                fitmask[wfit]=True
                                wwave=np.arange(min(wfit)-10,max(wfit)+10+1)
                                wmask[wwave]=True
           #Remove spectra if it doesn't actually get fit
            if np.sum(fitmask)==0:
                todel.append(name)
            else:
                #The following line is to find the indices of things in the wavelength array to include in the fit
                spec.w2fit=np.where(fitmask[wmask])[0] #spec.specw[w2fit] is wavelengths corresponding to specf array
                spec.specw=specw[wmask]
                spec.specf=spec.specf[fitmask]
                spec.spece=spec.spece[fitmask]                            
        for name in todel: del self.spectra[name]

            ##############################

    def plotTrans(self,zkey,ionkey,lkey,params=None,plotM=False,specname=None,dw=3):
        try:
            comp=self.components[zkey]
            ion=comp.ions[ionkey]
            line=ion.transitions[lkey]
        except KeyError:
            #print 'ion '+ionkey+' or line '+transkey+' not in model.'
            print("Transition not found in model. No plot made")
        fr0=line.fitregions[0]
        if specname is None: specname=fr0.specname
        spec=copy.deepcopy(self.spectra[specname])
        if fr0.wave:
            wavemin,wavemax=fr0.vmin,fr0.vmax
        else:
            w0=(1+comp.z)*line.wave0
            wavemin,wavemax=fr0.vminmax/(c/1e5)*w0-w0
                #                        specv = (specw-w0)/w0 * (c/1e5)
        w=(spec.specw > wavemin- dw) & (spec.specw < wavemax +dw)
        spec.specw=spec.specw[w]
        spec.specf=spec.specf[w]
        spec.spece=spec.spece[w]
        if lkey not in spec.lines: spec.lines.append(lkey)
        plt.plot(spec.specw,spec.specf,drawstyle='steps-mid')
        plt.ylim(-0.2,1.2)
        if plotM:
            vprof=vf.vpTau2Flux(vf.vpFromModel(self,spec),spec.kernel)
            plt.plot(spec.specw,vprof)
        elif params is not None:
            m2=self.list2Model(params)
            # m2=vf.list2Model(self,params)
            vprof=vf.vpTau2Flux(vf.vpFromModel(m2,spec),spec.kernel)
            plt.plot(spec.specw,vprof)

#################### Utility functions


def wave2v(waves,wave0,z):
    lambda0=wave0*(1+z)
    return (waves/lambda0-1)*c

    #This is a hack job since vfit.vplot doesn't look right for hires data
def vplot2(flux,err,wv,model,vp=None,fname='vplot2.pdf'):
    plotwaves=[]
    compkeys=model.components.keys()
    component=model.components[compkeys[0]]
    zc=component.z
    zs=compkeys
    vs=np.zeros(len(zs))
    for i in range(len(zs)):
        vs[i]=((1+zs[i])/(1+zc)-1)*c
    for ion in component.ions.values:
        for trans in ion.transitions.values():
            plotwaves.append(trans.wave0)
    nplot=len(plotwaves)
    dv=50
    fig=plt.figure(figsize=(5/0.875,1.618*5))
    for i in range(nplot):
        v=wave2v(wv,plotwaves[i],zc)
        w=np.logical_and(v>-dv-10,v<dv+10)
        plt.subplot(nplot,1,i+1)
        plt.step(v[w],flux[w],color='k',where='mid')
        plt.xlim=((-dv,dv))
        plt.ylim=(-0.1,1.2)
        for velo in vs:
            plt.axvline(velo,0.9,0.95)
    fig.subplots_adjust(hspace=0)
    plt.savefig(fname)



##################################

