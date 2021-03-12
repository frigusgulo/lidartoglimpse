import numpy as np 
import cupy 
import datetime
import sys
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as dist
from scipy.sparse.csc import csc_matrix as  sparsemat
from itertools import combinations
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

		self.initialize()

	def initialize(self, calibrate=False):
		#print(f"Initializing With {self.refscan}\n")
		self.particles = self.motionmodel.init_particles()
		self.particles_init = self.particles.copy()
		self.ref_index = np.linspace(0,len(self.particles_init)-1,len(self.particles_init)).astype(int)
		self.reference_points = self.refscan[self.refscan.query(self.motionmodel.xy,10),:]
		l = 0.1
		sigma2 = 0.01 #Observational Variance
		self.ref_mean = self.reference_points.mean(axis=0)
		self.ref_std = self.reference_points.std(axis=0)
		self.x_train = (self.reference_points - self.ref_mean)/self.ref_std
		K = np.exp(-dist(self.x_train[:,:2],self.x_train[:,:2],squared=True)/l**2)+ sigma2*np.eye(self.x_train.shape[0])
		self.Kinv = np.linalg.inv(K)
		self.n = self.motionmodel.n
		self.weights = np.ones(self.n)/self.n
		self.premu = self.Kinv @ self.x_train[:,2]

	def particle_mean(self)-> np.ndarray:
		"Weighted Particle Mean [x, y, z, vx, vy, vz]"
		return np.average(self.particles, weights=self.weights, axis=0)		
		
	'''
	def pointset_to_distmat(self,scan,pointset):
		start = time.time()
		pointset = np.array(pointset) 
		points = (scan.points[pointset] - self.ref_mean)/self.ref_std
		distances =  dist(points,points,squared=True).flatten()

		distances = np.exp(-distances/(0.2**2))
		maxind = max(pointset)+1
		pointsetinds = np.tile(pointset,(pointset.shape[0]))
		mat = sparsemat((distances,(pointsetinds,pointsetinds)),shape=(maxind,maxind) )
		print(f"\n KSS Took {time.time()-start} Seconds For {mat.shape} Matrix")
		return mat
'''
	
	def process_points(self,scan,point):
		return scan.query(point,radius=10)
		#pointset |= set(cloudset[-1])

	def scans_likelihood(self, scan: Scan,dt: datetime.timedelta=None):
		l = 0.1
		sigma2 = 0.01
		if dt is not None:
			self.dt = dt
		else:
			self.dt = scan.datetime - self.datetime


	
		self.datetime = scan.datetime
		self.particles = self.motionmodel.evolve_particles(self.particles,self.dt)
		delta_0 = self.particles_init[self.ref_index,:2] - self.motionmodel.xy
		delta_p = self.particles[:,:3] - self.particles_init[self.ref_index,:3] 
		# Will try and normalize by query point; using this approach precludes the use of precomputed values
		
		#pointset = set()
	
		start = time.time()
		testclouds = [scan.query(point,radius=10) for particle in list(self.particles[:,:2])]
		print(f"\n Test Clouds Queried in {time.time()-start} Seconds\n")
		#KSS = self.pointset_to_distmat(scan,pointset)
		#x_test = scan.points[np.array(pointset)]
		#x_test = (x_test - self.ref_mean)/self.ref_std

		#KSTAR = np.exp(-dist(x_test[:,:2],self.x_train[:,:2],squared=True)/l**2).flatten()
		#rows = np.tile(pointset,(self.x_train.shape[0]))
		#cols = np.tile(np.arange(self.x_train.shape[0]),len(pointset))
		#KSTAR = sparsemat((KSTAR,(rows,cols)),shape=(max(pointset)+1,self.x_train.shape[0]))

		particle_loglike = []
		start = time.time()
		for testcloud in testclouds:
			looptime = time.time()
			x_test = scan.points[np.array(testcloud)]
			x_test = (x_test - self.ref_mean)/self.ref_std

			Kss = np.exp(-dist(self.x_test[:,:2],self.x_test[:,:2],squared=True)/l**2)
			Kstar = np.exp(-dist(self.x_train[:,:2],self.x_test[:,:2],squared=True)/l**2)


		
			mu = Kstar @ self.premu
			Sigma = Kss - Kstar @ self.Kinv @ Kstar.T + sigma2*np.eye(Kss.shape[0])
			log_like = -0.5*(np.log(2*np.pi)*inds.shape[0] ) + np.linalg.slogdet(Sigma)[1] +0.5*(x_test[:,2]-mu)@np.linalg.inv(Sigma)@(x_test[:,2]-mu).T
			particle_loglike.append(log_like)
			D = time.time()
			print(f"\n Loop Time {D-looptime}\n")
		print(f"\n Likelihoods Took {time.time()-start} Seconds \n")
		particle_loglike = np.array(particle_loglike)
		w = np.exp(0.2*(particle_loglike-particle_loglike.max())) + 1e-300
		w/=w.sum()
		return w


	
	def update_weights(self,scan: Scan,dt: datetime.timedelta=None):
		self.weights = self.scans_likelihood(scan, dt) 


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
		initial_indexes = numpy.repeat(numpy.arange(self.n), repetitions)
		residuals = self.weights - repetitions
		residuals += 1e-300
		residuals *= 1 / residuals.sum()
		cumulative_sum = np.cumsum(residuals)
		cumulative_sum[-1] = 1.0
		additional_indexes = np.searchsorted(
		    cumulative_sum, np.random.random(self.n - len(initial_indexes))
		)
		indexes =  np.hstack((initial_indexes, additional_indexes))
		self.init_index = self.init_index[indexes]
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

			'''
			print(f"\n Variance X: {np.var(self.particles[:,0])} Variance Y: {np.var(self.particles[:,1])}")
			print(f"\n Variance VX: {np.var(self.particles[:,3])} Variance VY: {np.var(self.particles[:,4])}")
			x = self.particles[:,0] - np.mean(self.particles[:,0])
			y = self.particles[:,1]- np.mean(self.particles[:,1])
			vels = np.linalg.norm(self.particles[:,3:5],axis=-1)
			#colors = np.unique(vels)
			#print(vels)
			#vels = np.searchsorted(colors,vels)
			plt.xlim(-30,30)
			plt.ylim(-30,30)
			plt.title("Particle Velocities M/D")
			sc = plt.scatter(x,y,s=5,c=vels,vmin=-20,vmax=20,cmap=cm)
			plt.scatter(0,0,c='g',s=50)
			plt.colorbar(sc)
			plt.show()
			plt.clf()
			return self.particles
			#self.rewind()
			'''

		else:
			self.update_weights(scan)
			posterior = self.systematic_resample()
			#self.rewind()
			return [posterior,self.particles]