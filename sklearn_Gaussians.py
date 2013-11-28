# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:08:02 2013


"""
import db
import numpy as np
import pylab as pl
from sklearn import mixture
from utils import *
from scipy import linalg
import matplotlib as mpl

settingsFile = open('settings','r')
settings = eval(settingsFile.read())
dbDir = settings['dbDir']
n_samples = 300


tf_ids = db.dbUtils.getFromDB('califa_id', dbDir+'CALIFA.sqlite', 'mcmc_model2 where v22 > 50 and acceptanceFraction > 0.2 and v22 < 300')
print len(tf_ids)
  
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
x = db.dbUtils.getFromDB('v22', dbDir+'CALIFA.sqlite', 'mcmc_model2', ' where califa_id in '+califa_ids)
y = db.dbUtils.getFromDB('r_mag', dbDir+'CALIFA.sqlite', 'luminosity_errors2', ' where califa_id in '+califa_ids)
# generate random sample, two components
np.random.seed(0)
C = np.array([[0., -0.7], [3.5, .7]])
#X_train = np.r_[np.dot(np.random.randn(n_samples, 2), C),
          #np.random.randn(n_samples, 2) + np.array([20, 20])]

X = np.empty((x.shape[0], 2))
X[:, 0] = x
X[:, 1] = y

print X.shape

# Fit a mixture of gaussians with EM using five components
gmm = mixture.GMM(n_components=2, covariance_type='full')
gmm.fit(X)

#x = np.linspace(-20.0, 30.0)
#y = np.linspace(-20.0, 40.0)
# Fit a dirichlet process mixture of gaussians using five components
dpgmm = mixture.DPGMM(n_components=4, covariance_type='full')
dpgmm.fit(X)

color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                  (dpgmm, 'Dirichlet Process GMM')]):
    splot = pl.subplot(2, 1, 1 + i)
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        pl.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=10, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        #ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        #ell.set_clip_box(splot.bbox)
        #ell.set_alpha(0.5)
        #splot.add_artist(ell)

    #pl.xlim(-10, 10)
    #pl.ylim(-3, 6)
    pl.xticks(())
    pl.yticks(())
    pl.title(title)

pl.show()


