import numpy as np 
import datetime
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as dist
from scipy.ndimage.measurements import center_of_mass as gmean
from scipy.sparse.csc import csc_matrix as  sparsemat
from itertools import combinations
from sklearn.gaussian_process.kernels import Matern,RationalQuadratic
cm = plt.cm.get_cmap('RdYlBu')
from .motion import CartesianMotion
from .scanset import Scanset
from .raster import Raster
from .filepath import Filepath
from .scan import Scan
from .scanset import Scanset 
import time
import cv2
import glimpse
import pdb
import sys

class Tracker:


	def __init__(
		self,
		motionmodel: CartesianMotion,
		initialscan: Scan = None,
		calibrate: bool = False):

		self.motionmodel = motionmodel
		self.refscan = initialscan
		self.datetime = self.refscan.datetime
		self.calibrate = calibrate
		self.particles = None
		self.weights = None
		self.refclusters = None

		self.scans_like = []

		self.weight_set = []
		self.particle_set = []
		self.posterior_set = []
		self.covariance_set = []
		self.timestep_set = []
		
		self.datetimes = []
		self.initialize()

	def optimal_radius(self,points=200):
		for i in np.arange(3,200,0.15).tolist():
			if max(self.refscan.query(self.motionmodel.xy,i,calibrate=True).shape)>=points:
				self.radius= i
				#print(f"Optimal Radius Is {self.radius} For {self.motionmodel.xy}")
				self.points=points
				break

	def normalize(self,array: np.ndarray,point: np.ndarray) -> np.ndarray:
		'''
		try:
			inds = np.random.choice(np.arange(array.shape[0]),size=np.rint(0.8*self.points),replace=True)
			array = array[inds,:]
		except:
			pass
		'''
		mean = np.median(array,axis=0)
		mean[:2] = point
		std = np.sqrt(np.mean((array-mean)**2,axis=0))
		array = (array-mean)/std

		elevs = np.abs(array[:,2])
		cutoff = np.percentile(elevs,60,axis=0)
		#cutoff = std[-1]
		keep = np.squeeze(np.argwhere(elevs < cutoff))
		array = array[keep,:]


		#size = int(max(array.shape))
		#inds = np.random.choice(np.arange(size),size=int(size//2),replace=False)
		return array






	def initialize(self, calibrate=False):
		self.optimal_radius()
		#self.length_scale = .17
		self.length_scale = 1/self.radius
		self.sigma2 = 2.5/self.radius
	
		self.kernel = Matern(length_scale=self.length_scale,nu=0.5) #MSE was about 3.4
		#self.kernel = RationalQuadratic(self.length_scale,self.alpha)
		self.particles = self.motionmodel.init_particles()
		self.particles_init = self.particles.copy()
		self.ref_index = np.linspace(0,len(self.particles_init)-1,len(self.particles_init)).astype(int)
		self.reference_points = self.refscan[self.refscan.query(self.motionmodel.xy,self.radius)]
		
		
		self.x_train = self.normalize(self.reference_points,self.motionmodel.xy)

		#print(f"Lengthscale : {self.length_scale} | Obsvar : {self.sigma2} | Alpha : {self.alpha}\n")
		K = self.kernel(self.x_train[:,:2])+ self.sigma2*np.eye(self.x_train.shape[0])
		self.Kinv = np.linalg.inv(K)
		
		self.n = self.motionmodel.n
		self.weights = np.ones(self.n)/self.n
		self.premu = self.Kinv @ self.x_train[:,2]

		self.posterior_set.append(self.particle_mean())
		self.weight_set.append(self.weights)
		self.particle_set.append(self.particles)
		self.covariance_set.append(self.particle_covariance())

	def re_initialize(self):
		prior = self.particle_mean()
		self.particles[:,:2] = self.motionmodel.xy
		self.reference_points = self.refscan[self.refscan.query(self.motionmodel.xy,self.radius)]
		self.x_train = self.normalize(self.reference_points,self.motionmodel.xy)
		K = self.kernel(self.x_train[:,:2])+ self.sigma2*np.eye(self.x_train.shape[0])
		self.Kinv = np.linalg.inv(K)
		self.premu = self.Kinv@self.x_train[:,2]

	
	def particle_mean(self)-> np.ndarray:
		"Weighted Particle Mean [x, y, z, vx, vy, vz]"
		return np.average(self.particles, weights=self.weights, axis=0)


	def particle_covariance(self) -> np.ndarray:
		"""Weighted (biased) particle covariance matrix (6, 6)."""
		return np.cov(self.particles.T,aweights=self.weights,ddof=0)

	def scans_likelihood(self, scan: Scan,dt: datetime.timedelta=None):
		if dt is not None:
			self.dt = dt
		else:
			self.dt = scan.datetime - self.datetime
			
		
			self.timestep_set.append(self.dt)
		self.datetime = scan.datetime
		self.particles = self.motionmodel.evolve_particles(self.particles,self.dt)
		delta_0 = self.particles_init[self.ref_index,:2] - self.motionmodel.xy
		delta_p = self.particles[:,:3] - self.particles_init[self.ref_index,:3] 
		# Will try and normalize by query point; using this approach precludes the use of precomputed values
		#unique_particles,indexes,rev_index = np.unique(np.rint(self.particles[:,:2]),axis=0,return_index=True,return_inverse=True)
		#print(f"Unique Reduction: {max(self.particles.shape)} ----> {max(indexes.shape)}",flush=True)
		#sys.stdout.write("\033[F") # Cursor up one line
		#time.sleep(.1)
		testclouds = [scan.query(point,self.radius) for point in list(self.particles[:,:2])]# - delta_0]
		particle_loglike = np.zeros_like(self.particles[:,0])
		start = time.time()
		self.counter = 0
		for i,testcloud in enumerate(testclouds):
			x_test_pre = scan.points[np.array(testcloud)]#- delta_p[i,:]
			#print(f"Normalization: {x_test_pre.shape} --> {x_test.shape}")
			
			if x_test_pre.shape[0] > self.points//13:
				x_test = self.normalize(x_test_pre,self.particles[i,:2])
				self.counter += 1
				Kss = self.kernel(x_test[:,:2])
				Kstar = self.kernel(x_test[:,:2],self.x_train[:,:2])
				mu = Kstar @ self.premu
				Sigma = Kss - Kstar @ self.Kinv @ Kstar.T + self.sigma2*np.eye(Kss.shape[0])
				log_like = -0.5*(np.log(2*np.pi)*x_test.shape[0] ) + np.linalg.slogdet(Sigma)[1] +0.5*(x_test[:,2]-mu)@np.linalg.inv(Sigma)@(x_test[:,2]-mu).T
				#pdb.set_trace()
				particle_loglike[i] = log_like
			else:
				#print(f"Not enough points: {x_test.shape}")
				particle_loglike[i] = np.nan

	
		particle_loglike[np.isnan(particle_loglike)] = np.nanmean(particle_loglike)
		w = np.exp((particle_loglike-particle_loglike.max()))
		w+= 1e-300
		w/=w.sum()
		
		return w



	def systematic_resample(self):

		positions = (np.arange(self.n) + np.random.random() ) * (1/(self.n))
		cumulative_weight = np.cumsum(self.weights)
		cumulative_weight[-1] = 1
		indexes = np.searchsorted(cumulative_weight,positions)
		self.particles = self.particles[indexes,:]
		self.weights = self.weights[indexes]
		self.weights *= 1/self.weights.sum()
		self.ref_index = self.ref_index[indexes]

		posterior = self.particle_mean()
		return posterior


	def residual_resample(self):
		repetitions=(self.n*self.weights).astype(int)
		initial_indexes = np.repeat(np.arange(self.n), repetitions)
		residuals = self.weights - repetitions
		residuals += 1e-300
		residuals *= 1 / residuals.sum()
		cumulative_sum = np.cumsum(residuals)
		cumulative_sum[-1] = 1.0
		additional_indexes = np.searchsorted(
		    cumulative_sum, np.random.random(self.n - len(initial_indexes))
		)
		indexes =  np.hstack((initial_indexes, additional_indexes))
		self.ref_index = self.ref_index[indexes]
		self.particles = self.particles[indexes,:]
		self.weights = self.weights[indexes]
		self.weights *= 1/self.weights.sum()
		posterior = self.particle_mean()
		return posterior

	


	def track(self, observation ,likelihoods = None,dt: datetime.timedelta =None):

		if isinstance(observation,glimpse.Image) and likelihoods is not None:
			#self.image_like.append(likelihoods)
			self.weights = likelihoods

		elif isinstance(observation,Scan) and likelihoods is None:
			self.weights = self.scans_likelihood(observation)

	


		self.weight_set.append(self.weights)
		posterior = self.systematic_resample()
		self.datetimes.append(observation.datetime)
		self.posterior_set.append(posterior)
		self.particle_set.append(self.particles)
		self.covariance_set.append(self.particle_covariance())
