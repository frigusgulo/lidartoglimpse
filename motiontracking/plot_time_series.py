'''
This script is used to plot a time series of estimated velocities along with any other 
associated events and observations
'''

import numpy as np 
import datetime
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as dist
from scipy.ndimage.measurements import center_of_mass as gmean
from scipy.sparse.csc import csc_matrix as  sparsemat
from itertools import combinations
from sklearn.gaussian_process.kernels import Matern,RationalQuadratic,RBF
from sklearn.neighbors import NearestNeighbors

cm = plt.cm.get_cmap('RdYlBu')
from .motion import CartesianMotion
from .scanset import Scanset
from .raster import Raster
from .filepath import Filepath
from .scan import Scan
from .scanset import Scanset 
import time
import cv2

'''
INPUTS: Tracker {posteriors,particles,covariances,weights}
		Observations: {scans,photos}
		Environmental : {met station data}
		Events: {calving,plume}


OUTPUTS: Plot: Dates along x axis 
	- Median filtered velocities, use the median velocities from the point and its 4 nearest neighbors. 
	 plot color should be a function of distance from the centerline.
	
	 Median filtered uncertainty, use the magnitude of the median weighted  
	 standard deviations from the x and y velocities from the point and its 4 nearest neighbors
	 (use "fill between" to plot uncertainty)
	** Should plot uncertainty of evolved particles prior to updating with a given observation

	Observations - Plot vertical line at observations, with color dependent on observation instrument (i.e. blue for camera 1, red for cam 2, green for scan)

	Events - Plot vertical line corresponding to an event (i.e. calving)

	Environmental - Plot relevant environmental data with in plots below velocity estimates, along with correlation of velocity estimates
					(units on left y axis, correlation on right. Dates are x axis)



'''

class timeseriesplotter():

	def __init__(self,
		posteriors: np.ndarray,
		covariances: np.ndarray,
		observations: list[tuple],
		updates: np.ndarray = None,
		lengthscale=12): -> None


    '''
    	Posteriors [tracker,state,time]

    	particles [tracker,particles,time]

    	covariances [tracker, covariance,time]

    '''
    self.kernel = RBF(lengthscale)
	self.posteriors = posteriors
	self.covariances = covariances
	if updates:
		self.updates = matplotlib.dates.date2num(updates)

	self.observations = observations

	self.observations[:,0] = np.squeeze(matplotlib.dates.date2num(self.observations[:,0]))
	self.observations[self.observations[:,0].argsort(),:]


	def timespan(self,key=0):
		'''
		Get observational timespan to plot as defined by an observational key. 
		0 corresponds to scans
		'''

		self.minimum = [np.min(self.observations[:,self.observations[:,1]==key])]
		self.maximum = [np.max(self.observations[:,self.observations[:,1]==key])]
		self.span = self.maximum[0] - self.minimum[0]

		self.minimum.append(np.argmin(self.observations[:,0]>=self.minimum[0]))
		self.maximum.append(np.argmax(self.observations[:,0]<=self.maximum[0]))

		self.observations = self.observations[self.minimum[1]:self.maximum[1]+1,:]
		if self.updates:
			self.bounds = [(self.updates>=self.minimum[0]) & (self.updates<=self.maximum[0])]
		else:
			self.bounds = np.s_[self.minimum[1]:self.maximum[1]+1]
		print(f"Observation timespan: {self.span}\n")



	def velocities(self):
		# take the 4 nearnest neighbors and compute the median velocity magnitude

		posteriors = self.posteriors[:,:,self.bounds]
		self.filtered_velocities = np.zeros((posteriors.shape[0],3,posteriors.shape[-1]))
		for t in range(posteriors.shape[-1]):
			neighbors =  NearestNeighbors(n_neighbors=4).fit(posterior[:,:2,t])
			for i,posterior in enumerate(posteriors[:,:,t].tolist()):
				dist,inds = neighbors.kneighbors(posterior[:2])
				inds = np.hstack((inds,i))
				weighted_posterior = np.median(posterior[inds,:],axis=0)
				velocity = np.sqrt(weighted_posterior[3]**2 + weighted_posterior[4]**2)
				filtered_post = np.hstack((posterior[:2],velocity))
				self.filtered_velocities[i,:,t] = filtered_post

	def variances(self):
		if self.filtered_velocities:
			covariances = self.covariances[:,:,:,self.bounds]
			self.filtered_variances = np.zeros((self.filtered_velocities.shape[0],covariances.shape[-1]))
			for t in range(covariances.shape[-1]):
				neighbors =  NearestNeighbors(n_neighbors=4).fit(self.posteriors[:,:2,t])
				for i,cov_mat in enumerate(covariances[:,:,t].tolist()):
					dist,inds = neighbors.kneighbors(self.posteriors[i,:2,t])
					inds = np.hstack((inds,i))
					variances = [np.diag(mat) for mat in covariances[inds,:,:,t].tolist()]
					mag = [np.sqrt(variance[3] + variance[4]) for variance in variances]

					self.filtered_variances[i,t] = mag 
		else:
			raise Error("filtered velocities not computed")

	def interp(self,data,interpbounds):
		# TODO
		# Interpolate data between observations given the (time,data) and (interp values)
		time = np.linspace(interpbounds[0],interpbounds[1],24*(interpbounds[0]-interpbounds[1])) #hourly interpolation
		Kinv = np.linalg.inv(self.kernel(data[:,0]) + 0.2*np.eye(data[:,0].shape[0]))
		KS = self.kernel(time,data[:,0])
		KSS = self.kernel(time)

		mu = KS@Kinv@data[:,1]

		sigma = KSS - KS @Kinv@KS.T + 0.2*np.eye(KSS.shape[0])

		return (time,mu,sigma)

	def process_environmentals(self,data,label):
		'''
		data format : (datatime.datetime, value)
		'''
		data[:,0] = np.squeeze(matplotlib.dates.date2num(data[:,0]))




if __name__ == "__main__":

	covs = np.load("data/covariances",covariances)
	posts = np.load("data/tracks",tracks)
	parts = np.load("data/particles",particles)
	timesteps = np.load("data/timesteps",timesteps)
	errors = np.load("data/error",error)
	radii = np.load("data/radii",radii_sigma2)



		


