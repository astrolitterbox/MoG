from __future__ import division
import numpy as np
from utils import *
import matplotlib as mpl
mpl.use('Agg')
import db
import math
import matplotlib.pyplot as plt
import sampler, ensemble
from adict import *
from parse_cfg import *
import scipy as sc
import scipy.linalg as linalg
from matplotlib.patches import Ellipse

#Settings
#mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 10

defaults = parse_config('default.cfg')
opt = parse_config('mcmc_options_tf.cfg', defaults)
Npar = 2


def Linear2dModel(params, data):
  x, y, xerr, yerr = data
  z = np.zeros((len(y), 2))
  ycovar= np.zeros((2,len(y),2))
  ycovar[0,:,0]= yerr**2.
  ycovar[1,:,1]= y**2.
  ycovar[0,:,1]= xerr*np.sqrt(ycovar[0,:,0]*ycovar[1,:,1])
  ycovar[1,:,0]= ycovar[0,:,1]
  slope, offset = params
  

def TFmodel(params, data):  
  x, y, xerr, yerr = data
  slope, offset = params
  model_y = np.log10((x**slope) * 10**offset)
  return model_y
  
def LogTFmodel(params, data):
  x, y, xerr, yerr = data
  slope, offset = params
  model_y = slope*x + offset
  return model_y

def logTFResiduals(params, data):
  x, y, xerr, yerr = data
  slope, offset = params  
  N = y.shape[0]
  return np.sum((1/N)*((y - LogTFmodel(params, data))**2/(xerr**2 + slope**2 * yerr**2))) 

def getPrior(params):
    if (abs(params[0]) > 20 or abs(params[1] > 20)):
            return -np.inf
    else:
            return 0

def TFresiduals1Err(params, data):
  x, y, xerr, yerr = data
  return np.sum(((y - TFmodel(params, data))/yerr)**2)
  

def lnProb(params, data):
   x, y, xerr, yerr = data
   #model = TFmodel(params, data)
   #resid = TFresiduals1Err(params, data)
   resid = logTFResiduals(params, data)
   prior = getPrior(params)
   lnl = -0.5*((resid))
   return lnl + prior

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



def fit_TF():
  tf_ids = db.dbUtils.getFromDB('califa_id', dbDir+'CALIFA.sqlite', 'mcmc_model2', ' where v22 > 50')
  
  #bad_ids = [7, 20, 73, 100, 153, 119, 127, 515, 608, 657, 665, 676, 764, 824, 841, 856, 866, 872, 891, 935, 937, 938, 4, 17, 33, 475, 518, 548, 577, 680, 798, 827, 828, 829, 833, 840, 841, 845, 846, 851, 859, 860, 861, 864, 885, 888, 891, 892, 893, 894, 895, 898, 900, 903, 905] #visual outliers added afdter 939, remove
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
  '''
  d = np.genfromtxt("data/derived_TFR.csv", delimiter=",")
  absMag_d = d[:, 0]
  vVir_lo = d[:, 1]
  vVir = d[:, 2]
  vVir_hi = d[:, 3]
  vVir_c = d[:, 4]
  '''
 
 
  

  lambda_ids = db.dbUtils.getFromDB('califa_id', dbDir+'CALIFA.sqlite', 'lambda_incl', ' where califa_id in '+califa_ids)  
  lambda_ids = sqlify(lambda_ids)
  absMag = db.dbUtils.getFromDB('r_mag', dbDir+'CALIFA.sqlite', 'luminosity_errors2', ' where califa_id in '+lambda_ids)
  absMagErr = np.abs(db.dbUtils.getFromDB('lower_error', dbDir+'CALIFA.sqlite', 'luminosity_errors2', ' where califa_id in '+lambda_ids))
  vel_22 = db.dbUtils.getFromDB('v22', dbDir+'CALIFA.sqlite', 'mcmc_model2', ' where califa_id in '+lambda_ids)
  vel_22 = np.abs(vel_22)
  vel_err = db.dbUtils.getFromDB('err_vc', dbDir+'CALIFA.sqlite', 'mcmc_model2', ' where califa_id in '+lambda_ids)
  lambda_par = db.dbUtils.getFromDB('lambda_incl', dbDir+'CALIFA.sqlite', 'lambda_incl', ' where califa_id in '+lambda_ids) 
  log_vel = np.log10(vel_22)
  vel_err_u = np.abs(log_vel - np.log10(vel_err+vel_22))
  vel_err_l = np.abs(log_vel - np.log10(np.abs(vel_22 - vel_err)))
  mean_err = (vel_err_u + vel_err_l)/2
  ba = db.dbUtils.getFromDB('ba', dbDir+'CALIFA.sqlite', 'bestba', ' where califa_id in '+lambda_ids)  
 
  
  C = 0.95 #bandpass - dependent internal extinction coefficient
  R0 = 0.418 #correction to 70 deg
  Rmax = 0.7
  Rmin = 0.27
  
  ecc = -1*np.log10(1 - ba)
  A = np.zeros((absMag.shape))
  A = ecc - R0
  A[np.where(ecc < 0.27)] = Rmin - R0
  A[np.where(ecc > 0.7)] = Rmax - R0  
  absMag = absMag + A  
  
  print absMag.shape, mean_err.shape, vel_22.shape


  #absMag = np.reshape(absMag, (absMag.shape[0], 1))
  #log_vel = np.reshape(log_vel, (log_vel.shape[0], 1))   
  #absMagErr = np.reshape(absMagErr, (absMagErr.shape[0], 1))
  #mean_err = np.reshape(mean_err, (mean_err.shape[0], 1))  
  data = [absMag, log_vel, absMagErr, mean_err]
    
  initParams = [-5, 8]
  print "Running MCMC with %i steps" % opt.Nmcmc
  p0 = [getRandParams(initParams) for i in xrange(opt.Nwalkers)]   
  sampl = ensemble.EnsembleSampler(opt.Nwalkers, Npar, lnProb, args=[data], threads=opt.Nthreads)
  pos, prob, state = sampl.run_mcmc(p0, opt.Nburn)
  sampl.reset()
  sampl.run_mcmc(pos, opt.Nmcmc, rstate0=state)
  acceptanceFraction = np.mean(sampl.acceptance_fraction)
  print("Mean acceptance fraction:", acceptanceFraction)
  print("Autocorrelation time:", sampl.acor)
  
  slope = np.mean(sampl.flatchain[:,0])
  offset = np.mean(sampl.flatchain[:,1])
  
  
  slope_err = np.std(sampl.flatchain[:,0])
  offset_err = np.std(sampl.flatchain[:,1])
  print slope, offset, 'params', slope_err, offset_err
  
  
  #---------------------------- Fast or slow rotators -----


  #TIDAL
  #TF by lambda parameter -- kin_params table
  
  #lambda_par = np.reshape(lambda_par, (lambda_par.shape[0], 1))
  
  print lambda_par, lambda_par.shape
  
  fitlogx = np.linspace(2.0, 2.55)
  fit_line = 1/slope*(fitlogx) - offset/slope
  fig = plt.figure()
  ax2 = fig.add_subplot(111) 
  plt.title("Slope: "+str(round(1/slope, 2))+" offset: "+str(round(-offset/slope, 2)))  
  ax2.axis([1.6, 2.7, -18, -24.2]) 
  ax2.errorbar(log_vel, absMag, xerr=mean_err, yerr=absMagErr, linestyle = "none", color="black")
    
  plt.plot(fitlogx, fit_line, c = "black", linewidth = 3)
  cb = plt.scatter(log_vel, absMag, c=lambda_par, s=60, edgecolor="none", label="CALIFA", zorder=50, alpha=0.85)
  cbar = plt.colorbar(cb)
  plt.xlabel(r"$log(v_{2.2})$", fontsize=18)
  plt.ylabel(r"$M_{r}$", fontsize=18)
  cbar.set_label('lambda parameter')
  #plt.legend(loc='upper left', scatterpoints = 1)
  plt.savefig("img/tf/TF_lambda")      
 
  # ------------------fitting FRs only ---------------------

  fast = np.where(lambda_par > 0.7)
  print fast[0].shape

  data = [absMag[fast], log_vel[fast], absMagErr[fast], mean_err[fast]]
    
  initParams = [-5, 8]
  print "Running MCMC with %i steps" % opt.Nmcmc
  p0 = [getRandParams(initParams) for i in xrange(opt.Nwalkers)]   
  sampl = ensemble.EnsembleSampler(opt.Nwalkers, Npar, lnProb, args=[data], threads=opt.Nthreads)
  pos, prob, state = sampl.run_mcmc(p0, opt.Nburn)
  sampl.reset()
  sampl.run_mcmc(pos, opt.Nmcmc, rstate0=state)
  acceptanceFraction = np.mean(sampl.acceptance_fraction)
  print("Mean acceptance fraction:", acceptanceFraction)
  print("Autocorrelation time:", sampl.acor)
  
  slope = np.mean(sampl.flatchain[:,0])
  offset = np.mean(sampl.flatchain[:,1])
  
  
  slope_err = np.std(sampl.flatchain[:,0])
  offset_err = np.std(sampl.flatchain[:,1])
  print slope, offset, 'params', slope_err, offset_err  
  
  fitlogx = np.linspace(2.0, 2.55)
  fit_line = 1/slope*(fitlogx) - offset/slope
  fig = plt.figure()
  ax2 = fig.add_subplot(111) 
  plt.title("Slope: "+str(round(1/slope, 2))+" offset: "+str(round(-offset/slope, 2)))  
  ax2.axis([1.6, 2.7, -18, -24.2]) 
  ax2.errorbar(log_vel[fast], absMag[fast], xerr=mean_err[fast], yerr=absMagErr[fast], linestyle = "none", color="black")
    
  plt.plot(fitlogx, fit_line, c = "black", linewidth = 3)
  cb = plt.scatter(log_vel[fast], absMag[fast], c=lambda_par[fast], s=60, edgecolor="none", label="CALIFA", zorder=50, alpha=0.85)
  cbar = plt.colorbar(cb)
  plt.xlabel(r"$log(v_{2.2})$", fontsize=18)
  plt.ylabel(r"$M_{r}$", fontsize=18)
  cbar.set_label('lambda parameter')
  #plt.legend(loc='upper left', scatterpoints = 1)
  plt.savefig("img/tf/TF_FRs") 
fit_TF()

  
