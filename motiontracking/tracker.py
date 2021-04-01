import numpy as np 
import datetime
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as dist
from scipy.ndimage.measurements import center_of_mass as gmean
from scipy.sparse.csc import csc_matrix as  sparsemat
from itertools import combinations
from sklearn.gaussian_process.kernels import Matern
cm = plt.cm.get_cmap('RdYlBu')
from .motion import CartesianMotion
from .scanset import Scanset
from .raster import Raster
from .filepath import Filepath
from .scan import Scan
from .scanset import Scanset 
import time

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
		self.testdem = Raster("/home/dunbar/Research/helheim/data/2016_cpd_vels/160819_060211.tif")
		

		self.weight_set = []
		self.particle_set = []
		self.posterior_set = []
		self.covariance_set = []
		self.timestep_set = []
		self.initialize()

	def optimal_radius(self,points=120):
		for i in np.linspace(5,21,20).tolist():
			if max(self.refscan.query(self.motionmodel.xy,i,calibrate=True).shape)>=points:
				self.radius= i
				print(f"Optimal Radius Is {self.radius} For {self.motionmodel.xy}\n")
				self.points=points
				break

	def normalize(self,array: np.ndarray) -> np.ndarray:
		mean = np.median(array,axis=0)
		std = np.mean(array-mean,axis=0)
		elevs = np.abs(array[:,2] - mean[2])
		cutoff = np.percentile(elevs,40,axis=0)
		keep = np.squeeze(np.argwhere(elevs < cutoff))
		array = array[keep,:]
		return (array-mean)/std

	def initialize(self, calibrate=False):
		self.timecounter=0
		self.optimal_radius()
		length_scale = self.radius/70
		self.kernel = Matern(length_scale=length_scale,nu=5/2)
		self.particles = self.motionmodel.init_particles()
		self.particles_init = self.particles.copy()
		self.ref_index = np.linspace(0,len(self.particles_init)-1,len(self.particles_init)).astype(int)
		self.reference_points = self.refscan[self.refscan.query(self.motionmodel.xy,self.radius,maxpoints=self.points)]
		self.sigma2 = 0.06

		self.x_train = self.normalize(self.reference_points)

		K = self.kernel(self.x_train[:,:2])+ self.sigma2*np.eye(self.x_train.shape[0])
		self.Kinv = np.linalg.inv(K)
		
		self.n = self.motionmodel.n
		self.weights = np.ones(self.n)/self.n
		self.premu = self.Kinv @ self.x_train[:,2]

		self.posterior_set.append(self.particle_mean())
		self.weight_set.append(self.weights)
		self.particle_set.append(self.particles)
		self.covariance_set.append(self.particle_covariance())

	def re_initialize(self,scan):
		print("Re-initializing")
		self.timecounter=0
		self.refscan = scan
		self.optimal_radius()
		length_scale = self.radius/70
		self.kernel = Matern(length_scale=length_scale,nu=5/2)
		prior = self.particles.copy()
		self.particles = self.motionmodel.init_particles()
		self.particles[:,3:] = prior[:,3:]
		self.ref_index = np.linspace(0,len(self.particles_init)-1,len(self.particles_init)).astype(int)
		self.reference_points = self.refscan[self.refscan.query(self.motionmodel.xy,self.radius,maxpoints=self.points)]
		self.x_train = self.normalize(self.reference_points)
		K = self.kernel(self.x_train[:,:2])+ self.sigma2*np.eye(self.x_train.shape[0])
		self.Kinv = np.linalg.inv(K)
		self.n = self.motionmodel.n
		self.premu = self.Kinv @ self.x_train[:,2]
	
	def particle_mean(self)-> np.ndarray:
		"Weighted Particle Mean [x, y, z, vx, vy, vz]"
		return np.average(self.particles, weights=self.weights, axis=0)


	def particle_covariance(self) -> np.ndarray:
		"""Weighted (biased) particle covariance matrix (6, 6)."""
		return np.cov(self.particles.T, aweights=self.weights, ddof=0)


	def scans_likelihood(self, scan: Scan,dt: datetime.timedelta=None):
		l = 0.17
		self.sigma2 = 0.06
		if dt is not None:
			self.dt = dt
		else:
			self.dt = scan.datetime - self.datetime
			
			self.timecounter += self.dt.total_seconds() / self.motionmodel.time_unit.total_seconds() 
			self.timestep_set.append(self.dt)
		self.datetime = scan.datetime
		self.particles = self.motionmodel.evolve_particles(self.particles,self.dt)
		delta_0 = self.particles_init[self.ref_index,:2] - self.motionmodel.xy
		delta_p = self.particles[:,:3] - self.particles_init[self.ref_index,:3] 
		# Will try and normalize by query point; using this approach precludes the use of precomputed values
		testclouds = [scan.query(point,self.radius,maxpoints=self.points) for point in list(self.particles[:,:2]) - delta_0]
		particle_loglike = []
		start = time.time()

		for i,testcloud in enumerate(testclouds):
			x_test = scan.points[np.array(testcloud)] - delta_p[i,:]
			x_test = self.normalize(x_test)
			Kss = self.kernel(x_test[:,:2])
			Kstar = self.kernel(x_test[:,:2],self.x_train[:,:2])
			mu = Kstar @ self.premu
			Sigma = Kss - Kstar @ self.Kinv @ Kstar.T + self.sigma2*np.eye(Kss.shape[0])
			log_like = -0.5*(np.log(2*np.pi)*x_test.shape[0] ) + np.linalg.slogdet(Sigma)[1] +0.5*(x_test[:,2]-mu)@np.linalg.inv(Sigma)@(x_test[:,2]-mu).T
			particle_loglike.append(log_like)

		particle_loglike = np.array(particle_loglike)
		w = np.exp(0.5*(particle_loglike-particle_loglike.max())) + 1e-300
		w/=w.sum()
		return w


	
	def update_weights(self,scan: Scan,dt: datetime.timedelta=None):
		self.weights = self.scans_likelihood(scan, dt)

		self.weight_set.append(self.weights)

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

	


	def track(self, scan: Scan=None,calibrate: bool = False,dt: datetime.timedelta =None):
		if calibrate:
			#print(f"\n Calibrating From {self.refscan}\n")
			
			self.update_weights(self.refscan,dt)
			posterior = self.systematic_resample()

		else:
			self.update_weights(scan)
			posterior = self.systematic_resample()

			if self.timecounter >= 4:
				self.re_initialize(scan)

			self.posterior_set.append(posterior)
			self.particle_set.append(self.particles)
			self.covariance_set.append(self.particle_covariance())
			#try:
			print("===================================")
			print(f"Posterior Velocity Vector: {self.posterior_set[-1][3:5]}\n")
			#print(f"Posterior Displacement: {(self.dt.total_seconds()/(3600*24))*np.linalg.norm(self.posterior_set[-1][:2] - self.posterior_set[-2][:2])}")
			print(f"Posterior Velocity: {np.sqrt(self.posterior_set[-1][3]**2 +self.posterior_set[-1][4]**2 )}")
			print(f"Test Velocity: {24*self.testdem.dem.read(1)[self.testdem.index(self.posterior_set[-2][:2])]}\n")
			print(f"Raidus: {self.radius}")
			#self.rewind()
			#except:
				#pass
			