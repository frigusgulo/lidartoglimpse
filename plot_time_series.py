'''
This script is used to plot a time series of estimated velocities along with any other 
associated events and observations
'''

import os
import numpy as np 
import datetime
import sys
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics.pairwise import euclidean_distances as dist
from scipy.ndimage.measurements import center_of_mass as gmean
from scipy.sparse.csc import csc_matrix as  sparsemat
from itertools import combinations
from sklearn.gaussian_process.kernels import Matern,RationalQuadratic,RBF
from sklearn.neighbors import NearestNeighbors
import matplotlib.cm as cm
from motiontracking.motion import CartesianMotion
from motiontracking.scanset import Scanset
from motiontracking.raster import Raster
from motiontracking.filepath import Filepath
from motiontracking.scan import Scan
from motiontracking.scanset import Scanset 
import time
import cv2
import pdb
import re
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
		observations: list=None,
		updates: np.ndarray = None,
		lengthscale=2): 



		self.kernel = RBF(lengthscale)
		self.posteriors = np.swapaxes(posteriors,1,-1)

		self.covariances = np.swapaxes(covariances,1,-1)
	
		self.updates = updates
		if updates is not None:
			self.updates = matplotlib.dates.date2num(updates)

		if observations is not None:
			self.observations = observations
			self.observations[:,0] = matplotlib.dates.date2num(self.observations[:,0])
		
			self.observations = np.unique(self.observations.astype(np.float),axis=0)

		self.datadict = dict()

	def timespan(self,key=0):
		'''
		Get observational timespan to plot as defined by an observational key. 
		0 corresponds to scans
		'''

		self.minimum = np.min(self.observations[:,key])
		self.maximum = np.max(self.observations[:,key])
		self.span = self.maximum - self.minimum
		

	
		print(f"Observation timespan: {np.rint(self.span)} Days\n")



	def velocities(self):
		# take the 4 nearnest neighbors and compute the median velocity magnitude

		#posteriors = self.posteriors

		self.variances = np.zeros((self.posteriors.shape[0],self.posteriors.shape[-1]))
		self.filtered_velocities = np.zeros((self.posteriors.shape[0],3,self.posteriors.shape[-1]))
		for t in range(self.posteriors.shape[-1]):
			neighbors =  NearestNeighbors(n_neighbors=8).fit(self.posteriors[:,:2,t])
			for i in range(self.posteriors[:,:,t].shape[0]):

		
				dist,inds = neighbors.kneighbors(self.posteriors[i,:2,t].reshape(1,-1))

				inds = np.vstack([np.array(inds),np.tile(i,max(inds.shape))]).T
				
			
				weighted_posterior = np.median(self.posteriors[inds[:,0],:,t],axis=0)
				self.variances[i,t] = np.median(self.getvariances(self.covariances[inds[:,0],:,:,t]),axis=0)
		
				
				velocity = np.sqrt(weighted_posterior[3]**2 + weighted_posterior[4]**2)
				filtered_post = np.hstack((weighted_posterior[:2],velocity))
				self.filtered_velocities[i,:,t] = filtered_post


	def getvariances(self,covariances):
		variances = []
		for i in range(covariances.shape[0]):
			var = np.diag(covariances[i,:,:])
			var = np.sqrt(var[3] + var[4])
			variances.append(var)

		return variances


	def interp(self,x,y,interpx):
		# TODO
		# Interpolate data between observations given the (time,data) and (interp values)
		#time = np.linspace(interpbounds[0],interpbounds[1],24*(interpbounds[0]-interpbounds[1]).days) #hourly interpolation
		x = x.reshape(-1,1)
		interpx = interpx.reshape(-1,1)
		Kinv = np.linalg.inv(self.kernel(x) + 0.1*np.eye(x.shape[0]))
		KS = self.kernel(interpx,x)
		KSS = self.kernel(interpx)
		#pdb.set_trace()
		mu = KS@Kinv@y
		sigma = KSS - KS @Kinv@KS.T + 0.1*np.eye(KSS.shape[0])
		return mu,sigma

	def interp_velocities(self,interpx):
		
		self.interp_vels = []
		self.interp_vars = []
		for t in range(self.filtered_velocities.shape[0]):
			#pdb.set_trace()
			mu, sig = self.interp(self.observations[:,0],self.filtered_velocities[t,-1,1:],interpx)
			mu2,sig2 = self.interp(self.observations[:,0],self.variances[t,1:],interpx)
			self.interp_vels.append(mu)
			self.interp_vars.append(mu2)

		self.interp_vels = np.array(self.interp_vels)
		self.interpx = interpx
		self.interp_vars = np.array(self.interp_vars)



	def process_environmentals(self,station,label,datax,datay,units,interpx):
		'''
		data format : (datatime.datetime, value)
		'''
		#pdb.set_trace()
		datax = matplotlib.dates.date2num(datax)
		#pdb.set_trace()
		keep = np.logical_and((datax>= self.minimum),(datax<=self.maximum))
		mu,sigma = self.interp(datax[keep],datay[keep].astype(np.float),interpx)
		#pdb.set_trace()
		try:
			self.datadict[station].update({label: {'mu':mu,'sigma': sigma,'units': units}})
		except KeyError:
			self.datadict[station]= {label: {'mu':mu,'sigma': sigma,'units': units}}

		#pdb.set_trace()

	def plot_vels(self):
		colors = cm.rainbow(np.linspace(0,1,self.filtered_velocities.shape[0]))
		self.filtered_velocities = np.sort(self.filtered_velocities,axis=0)
		
		for i in range(self.filtered_velocities.shape[0]):
			plt.plot_date(self.observations[:,0],self.filtered_velocities[i,-1,1:],color=colors[i],linestyle="-")
		plt.xlabel("Date")
		plt.ylabel("Velocity M/Day")
		plt.show()

	def init_plot(self,nrows,ncols):
		self.fig,self.ax = plt.subplots(nrows,ncols,squeeze=False)

	def plot_envs(self,plotlabels: list,ax,ylim):
		stations = self.datadict.keys()
		_base = len(plotlabels)
		colors = cm.rainbow(np.linspace(0,1,len(stations)*_base)) # seperate colors for each station/label pair
		plot_ind = 0
		for i,station in enumerate(self.datadict.keys()):
			for j,label in enumerate(self.datadict[station].keys()):
				if label in plotlabels:
						ax.plot_date(self.interpx,self.datadict[station][label]["mu"],color=colors[plot_ind],linestyle="solid",marker='None',linewidth="2.5",label=station+label)
						plot_ind += 1
						
		ax.set_xlabel("Time")
		ax.set_ylabel(plotlabels[0])
		ax.set_ylim(ylim[0],ylim[1])
		ax.grid(True)
		ax.legend()


	def plot_interp_vels(self,ax):
		days = matplotlib.dates.DayLocator()
		months = matplotlib.dates.MonthLocator()
		colors = cm.rainbow(np.linspace(0,1,self.filtered_velocities.shape[0]))
		self.interp_vels = np.sort(self.interp_vels,axis=0)
		for i in range(self.interp_vels.shape[0]):
			ax.plot_date(self.interpx,self.interp_vels[i,:],color=colors[i],linestyle="solid",marker='None',linewidth="2.5")
			#ax.fill_between(self.interpx,self.interp_vels[i,:]-self.interp_vars[i,:],self.interp_vels[i,:]+self.interp_vars[i,:],alpha=0.1,color=colors[i])
		meanvel = np.median(self.interp_vels,axis=0)
		meanvar = np.median(self.interp_vars,axis=0)
	
		ax.plot(self.interpx,meanvel,'k-',linewidth="5",label="Median Velocity")
		ax.xaxis.set_minor_locator(days)
		ax.fill_between(self.interpx,meanvel-meanvar,meanvel+meanvar,alpha=0.25,color='k')
		for i in range(self.observations.shape[0]):
			if self.observations[i,1] == 0:
				ax.axvline(self.observations[i,0],linewidth=1,color='g')
			else:
				ax.axvline(self.observations[i,0],linewidth=1,color='r')

		yticks = np.arange(0,35,2)
		ax.set_yticks(yticks)
		ax.grid(True)
		ax.set_xlabel("Date")
		ax.set_ylabel("Velocity M/Day")
		ax.set_title("Helheim Glacier Velocity")
		ax.legend()
	

if __name__ == "__main__":

	obs = np.load("data/observations.npy",allow_pickle=True)
	covs = np.load("data/covariances.npy",allow_pickle=True)
	posts = np.load("data/tracks.npy",allow_pickle=True)
	parts = np.load("data/particles.npy",allow_pickle=True)
	timesteps = np.load("data/timesteps.npy",allow_pickle=True)
	errors = np.load("data/error.npy",allow_pickle=True)
	radii = np.load("data/radii.npy",allow_pickle=True)

	datadir = "/home/dunbar/Research/helheim/data"
	awsMIT_data = np.vstack(np.load(os.path.join(datadir,"MIT_data.npy"),allow_pickle=True))
	awsTAS_data = np.vstack(np.load(os.path.join(datadir,"TAS_data.npy"),allow_pickle=True))
	tidaldata = np.load(os.path.join(datadir,"tasiilaq_tidal.npy"),allow_pickle=True)
	awsfields = ['AirPressure(hPa)','AirTemperature(C)','SurfaceTemperature(C)','WindSpeed(m/s)','SensibleHeatFlux(W/m2)','LatentHeatFlux(W/m2)',
                        'IceTemperature1(C)','IceTemperature2(C)', 'IceTemperature3(C)', 'IceTemperature4(C)','IceTemperature5(C)', 'IceTemperature6(C)', 
                         'IceTemperature7(C)']

	TSP = timeseriesplotter(posts,covs,obs)
	TSP.timespan()
	TSP.velocities()
	interp_axis = np.linspace(TSP.minimum,TSP.maximum,500)
	TSP.interp_velocities(interp_axis)

	for i,field in enumerate(awsfields):
		TSP.process_environmentals("TAS",field,awsTAS_data[:,0],awsTAS_data[:,i+1],re.search(r"\(\)",field),interp_axis)

	#for i,field in enumerate(awsfields):
			#TSP.process_environmentals("MIT",field,awsMIT_data[:,0],awsMIT_data[:,i+1],re.search(r"\(\)",field),interp_axis)
	print(np.max(tidaldata[:,0]))
	print(tidaldata[np.argmax(tidaldata[:,0]),1])
	TSP.process_environmentals("Tidal Data","tide",tidaldata[:,0],tidaldata[:,1],"Meters",interp_axis)
	#pdb.set_trace()
	TSP.init_plot(3,1)
	TSP.plot_interp_vels(TSP.ax[0][0])
	TSP.plot_envs(["tide"],TSP.ax[1][0],(-1,1))
	#TSP.plot_envs(['IceTemperature1(C)','IceTemperature2(C)', 'IceTemperature3(C)', 'IceTemperature4(C)','IceTemperature5(C)', 'IceTemperature6(C)', 
                         #'IceTemperature7(C)'],TSP.ax[2][0],(-20,10))
	TSP.plot_envs(['SurfaceTemperature(C)'],TSP.ax[2][0],(-20,10))
	plt.show()
	
	#TSP.velocities()
	#TSP.variances()
	#TSP.plot_vels()
		


