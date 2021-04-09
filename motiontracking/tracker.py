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
		
		self.templateimage = None
		self.image_like = []
		self.scans_like = []

		self.weight_set = []
		self.particle_set = []
		self.posterior_set = []
		self.covariance_set = []
		self.timestep_set = []
		self.error = []
		self.initialize()

	def optimal_radius(self,points=220):
		for i in np.arange(4,20,0.15).tolist():
			if max(self.refscan.query(self.motionmodel.xy,i,calibrate=True).shape)>=points:
				self.radius= i
				print(f"Optimal Radius Is {self.radius} For {self.motionmodel.xy}")
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
		cutoff = np.percentile(elevs,25,axis=0)
		#cutoff = std[-1]
		keep = np.squeeze(np.argwhere(elevs < cutoff))
		array = array[keep,:]


		#size = int(max(array.shape))
		#inds = np.random.choice(np.arange(size),size=int(size//2),replace=False)
		return array

	def extract_template(self,image,uv,xdim=10,ydim=10):
		return cv.cvtColor(image[uv-xdim//2:uv+xdim//2,uv-ydim//2:uv+ydim//2],cv.COLOR_BGR2GRAY)

	def normalize_img(self,img):
		mean = np.mean(img)
		var = np.var(img)
		return (img-mean)/var

	def image_init(self,image):
		loc = image.xyz_to_uv(self.particle_mean()[:2])
		templateimage = self.extract_template(image,loc)
		self.templateimage = self.normalize_img(img)
		self.image_datetime = image.datetime 

	def get_images(self,particles):
		self.test_images = []
		if self.templateimage:
			for particle in particles:
				loc = image.xyz_to_uv(particle[:3])
				image = self.normalize_img(self.extract_template(image,loc))
				self.test_images.append(image)

	def image_sse(self):
		if len(self.test_images) >0:
			return np.array([cv.matchTemplate(img,self.templateimage,cv.TM_SQDIFF) for img in self.test_images])
			self.test_images = []


	def image_likelihood(self,image):
		if self.templateimage:
			self.dt = image.datetime - self.datetime
			self.particles = self.motionmodel.evolve_particles(self.particles,self.dt)
			self.get_images(self.particles)
			image_sse = self.image_sse()
			w =  np.exp(-image_sse) + 1e-300
			w/=w.sum()
			self.image_like.append(w)
			return w
		else:
			self.image_init(image)

	def initialize(self, calibrate=False):
		self.timecounter=0
		self.optimal_radius()
		self.length_scale = .17
		self.sigma2 = 0.33
		self.alpha = 10
		#print(f"Lengthscale : {self.length_scale} | Obsvar : {self.sigma2} | Alpha : {self.alpha}")
		self.kernel = Matern(length_scale=self.length_scale,nu=0.5) #MSE was about 3.4
		#self.kernel = RationalQuadratic(self.length_scale,self.alpha)
		self.particles = self.motionmodel.init_particles()
		self.particles_init = self.particles.copy()
		self.ref_index = np.linspace(0,len(self.particles_init)-1,len(self.particles_init)).astype(int)
		self.reference_points = self.refscan[self.refscan.query(self.motionmodel.xy,self.radius,maxpoints=self.points)]
		
		
		self.x_train = self.normalize(self.reference_points,self.motionmodel.xy)

		print(f"Lengthscale : {self.length_scale} | Obsvar : {self.sigma2} | Alpha : {self.alpha}\n")
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
	
		prior = self.particle_mean()
		self.particles = self.motionmodel.init_particles(prior=prior)
		
		
		self.reference_points = self.refscan[self.refscan.query(self.motionmodel.xy,self.radius,maxpoints=self.points)]
		self.x_train = self.normalize(self.reference_points,self.motionmodel.xy)
		K = self.kernel(self.x_train[:,:2])+ self.sigma2*np.eye(self.x_train.shape[0])
		self.Kinv = np.linalg.inv(K)
		self.premu = self.Kinv @ self.x_train[:,2]
	
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
			
			self.timecounter += self.dt.total_seconds() / self.motionmodel.time_unit.total_seconds() 
			self.timestep_set.append(self.dt)
		self.datetime = scan.datetime
		self.particles = self.motionmodel.evolve_particles(self.particles,self.dt)
		delta_0 = self.particles_init[self.ref_index,:2] - self.motionmodel.xy
		delta_p = self.particles[:,:3] - self.particles_init[self.ref_index,:3] 
		# Will try and normalize by query point; using this approach precludes the use of precomputed values
		testclouds = [scan.query(point,self.radius,maxpoints=self.points) for point in list(self.particles[:,:2])]# - delta_0]
		particle_loglike = []
		start = time.time()
		self.counter = 0
		for i,testcloud in enumerate(testclouds):
			x_test_pre = scan.points[np.array(testcloud)] #- delta_p[i,:]
			#print(f"Normalization: {x_test_pre.shape} --> {x_test.shape}")
			
			if x_test_pre.shape[0] > self.points//11:
				x_test = self.normalize(x_test_pre,self.particles[i,:2])
				self.counter += 1
				Kss = self.kernel(x_test[:,:2])
				Kstar = self.kernel(x_test[:,:2],self.x_train[:,:2])
				mu = Kstar @ self.premu
				Sigma = Kss - Kstar @ self.Kinv @ Kstar.T + self.sigma2*np.eye(Kss.shape[0])
				log_like = -0.5*(np.log(2*np.pi)*x_test.shape[0] ) + np.linalg.slogdet(Sigma)[1] +0.5*(x_test[:,2]-mu)@np.linalg.inv(Sigma)@(x_test[:,2]-mu).T
				particle_loglike.append(log_like)
			else:
				#print(f"Not enough points: {x_test.shape}")
				particle_loglike.append(np.nan)

		particle_loglike = np.array(particle_loglike)
		particle_loglike[np.isnan(particle_loglike)] = np.nanmean(particle_loglike)
		w = np.exp((particle_loglike-particle_loglike.max()))

		w[np.isnan(w)] = 0
		w+= 1e-300
		w/=w.sum()
		
		return w


	
	def update_weights(self):
	
		weights = self.scans_like.pop(-1)
		try:
			weights += self.image_like.pop(-1)
		except:
			pass
		weights /= weights.sum()

		self.weights = weights
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

	


	def track(self, scan: Scan=None,image=None,calibrate: bool = False,dt: datetime.timedelta =None):
		if calibrate:
			#print(f"\n Calibrating From {self.refscan}\n")
			
			self.update_weights(self.refscan,dt)
			posterior = self.systematic_resample()

		else:
			if image is not None:
				self.image_likelihood(image)
			if scan is not None:
				self.scans_like.append(self.scans_likelihood(scan))
				self.update_weights()

			posterior = self.systematic_resample()

			if self.timecounter >= 4:
				self.re_initialize(scan)

			self.posterior_set.append(posterior)
			self.particle_set.append(self.particles)
			self.covariance_set.append(self.particle_covariance())
			try:
				print("===================================")
				#print(f"Posterior Velocity Vector: {self.posterior_set[-1][3:5]}\n")
				#print(f"Posterior Displacement: {(self.dt.total_seconds()/(3600*24))*np.linalg.norm(self.posterior_set[-1][:2] - self.posterior_set[-2][:2])}")
				postvel = np.sqrt(self.posterior_set[-1][3]**2 +self.posterior_set[-1][4]**2 )
				testvel = 24*self.testdem.dem.read(1)[self.testdem.index(self.posterior_set[-2][:2])]

				print(f"Posterior Velocity: {postvel}\n")
				print(f"Test Velocity: {testvel}\n")
				print(f"Obsvar {self.sigma2}")
				print(f"Raidus: {self.radius}")
				#print(f"Effective Particles {100*(self.counter/self.n)}")
				#print(f"Covariance: {self.particle_covariance()}")
				#self.rewind()
				self.error.append( (postvel-testvel)**2)
				print(f"Root Squared Error: {np.sqrt(self.error[-1])}\n")
				print(f"Weight variance: {np.var(self.weights)}\n")			
			except:
				pass
			