# This code is filled with expressions that blow up at theta=0.5*pi;
# these can be replaced with much better expressions that make use of
# linear algebra; this is left as an exercise to the reader!

import math as ma
import matplotlib
matplotlib.use('Agg')
# pylab must go before numpy.random import
from pylab import *
import numpy as np
import numpy.random as random
from generate_data_backup import read_data
from matplotlib import rcParams
import scipy.linalg as linalg
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
import sampler, ensemble
import db
from utils import *

def logTFResiduals(params, data):
  x, y, xerr, yerr = data
  slope, offset = params  
  N = y.shape[0]
  return np.sum((1/N)*((y - LogTFmodel(params, data))**2/(xerr**2 + slope**2 * yerr**2))) 


def getPrior(params):
    if (abs(params[0]) > 20 or abs(params[1]) > 20):
            return -np.inf
    else:
            return 0

def lnProb(params, data):
   x, y, xerr, yerr = data
   resid = logTFResiduals(params, data)
   prior = getPrior(params)
   lnl = -0.5*((resid))
   return lnl + prior
  
def LogTFmodel(params, data):
  x, y, xerr, yerr = data
  slope, offset = params
  model_y = slope*x + offset
  return model_y

def getRandParams(initParams):
	out = []
	for i, x in enumerate(initParams):
		if i == 0: #vc
			out.append((np.random.normal(loc=1., scale=0.1)*x))
		elif i == 1: #c
			out.append(np.random.normal(loc=1., scale=0.1)*x)
		
		elif i == 4: #incl
			out.append(np.radians(np.random.normal(loc=1., scale=0.1)*np.degrees(x)))
			
		else:
			out.append((np.random.normal(loc=1., scale=0.3)*x))
	return out

tf_ids = db.dbUtils.getFromDB('califa_id', dbDir+'CALIFA.sqlite', 'mcmc_2')
bad_ids = []
GoodTF_ids = []
for i, gal in enumerate(tf_ids):
    if gal in bad_ids:
      print 'aaa'
      continue
    else:
      GoodTF_ids.append(gal)
print len(GoodTF_ids), 'no. of galaxies'    
califa_ids = sqlify(GoodTF_ids)
 
vel_22 = db.dbUtils.getFromDB('v22', dbDir+'CALIFA.sqlite', 'mcmc_2', ' where califa_id in '+califa_ids)
vel_22 = np.abs(vel_22)
vel_err = db.dbUtils.getFromDB('err_vc', dbDir+'CALIFA.sqlite', 'mcmc_2', ' where califa_id in '+califa_ids)
log_vel = np.log10(vel_22)
vel_err_u = np.abs(log_vel - np.log10(vel_err+vel_22))
vel_err_l = np.abs(log_vel - np.log10(np.abs(vel_22 - vel_err)))
mean_err = 0.7*(vel_err_u + vel_err_l)/2

print mean_err

absMag = db.dbUtils.getFromDB('r_mag', dbDir+'CALIFA.sqlite', 'luminosity_errors2', ' where califa_id in '+califa_ids)
absMagErr = np.abs(db.dbUtils.getFromDB('lower_error', dbDir+'CALIFA.sqlite', 'luminosity_errors2', ' where califa_id in '+califa_ids))

#inputData = np.genfromtxt('data_allerr.dat', delimiter=",")
#ids =  inputData[:, 0]
#log_vel = inputData[:, 1]
#abs_mag = inputData[:, 2]
#abs_mag_err = inputData[:, 3]
#vel_err = inputData[:, 4]
'''
N = len(GoodTF_ids)
C = np.zeros((N, N))
Y = log_vel
A = np.ones((N, 2))
A[:, 1] = absMag
yvar= np.zeros((N,2, 2))
ellipses = []

#Calculate the eigenvalues and the rotation angle
yvar[:,0,0]= absMagErr**2.
yvar[:,1,1]= absMag**2.
yvar[:,0,1]= vel_err*sqrt(yvar[:,0,0]*yvar[:,1,1])
yvar[:,1,0]= yvar[:,0,1]

for i in range(0, 1):
	eigs= linalg.eig(yvar[i,:,:])
	angle= arctan(-eigs[1][0,1]/eigs[1][1,1])/pi*180.
	thisellipse= Ellipse(array([log_vel[i],absMag[i]]),2*sqrt(eigs[0][0]),
							 2*sqrt(eigs[0][1]),angle)
	ellipses.append(thisellipse)
'''
#MCMC options
Nwalkers = 600
Nburn = 100
Nthreads = 2
Nmcmc = 300
Npar = 2

#init
data = [absMag, log_vel, absMagErr, mean_err]
    
initParams = [-5, 8]
print "Running MCMC with %i steps" % Nmcmc
p0 = [getRandParams(initParams) for i in xrange(Nwalkers)]   
sampl = ensemble.EnsembleSampler(Nwalkers, Npar, lnProb, args=[data], threads=Nthreads)
pos, prob, state = sampl.run_mcmc(p0, Nburn)
sampl.reset()
sampl.run_mcmc(pos, Nmcmc, rstate0=state)
acceptanceFraction = np.mean(sampl.acceptance_fraction)
print("Mean acceptance fraction:", acceptanceFraction)
print("Autocorrelation time:", sampl.acor)

slope = np.mean(sampl.flatchain[:,0])
offset = np.mean(sampl.flatchain[:,1])
print 'slope:', slope, 'offset:', offset

#fit_line = LogTFmodel((slope, offset), data)

#plotting
fig = plt.figure()
ax2 = fig.add_subplot(111) 
plt.title("Slope: "+str(round(slope))+" offset: "+str(round(offset, 2)))
fitlogx = np.linspace(2.0, 2.55)
fit_line = 1/slope*(fitlogx) - offset/slope
#ax2.axis([1.6, 2.6, -18, -24.2]) 
ax2.errorbar(log_vel, absMag, xerr=mean_err, yerr=absMagErr, linestyle = "none", color="black")  
ax2.plot(fitlogx, fit_line, c = "Crimson", linewidth = 2, label="Best fit") 
ax2.scatter(log_vel, absMag, c='RoyalBlue', marker='s', s=80, edgecolor="k", label="CALIFA") 
plt.xlabel(r"$log(v_{2.2})$", fontsize=18)
plt.ylabel(r"$M_{r}$", fontsize=18)
plt.legend(loc='upper left', scatterpoints = 1)
plt.savefig("TFR")
