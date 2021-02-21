import numpy as np 
import datetime
import sys
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances as dist
cm = plt.cm.get_cmap('RdYlBu')
from .motion import CartesianMotion
from .scanset import Scanset
from .raster import Raster
from .filepath import Filepath
from .scan import Scan, Particleset
from .scanset import Scanset 

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

		#self.reference_point = np.array([530884.00,7356524.00]) 
		self.initialize()

	def query_scan(self,scan: Scan,loc):
		points = scan.radialcluster(point=loc,radius=20)

		return points

	
	def getfeatures(self,particles,scan: Scan):
		features = [self.query_scan(scan,particles[i,:2]) for i in range(self.particles.shape[0])]
		return np.array(features)

	
	def initialize(self, calibrate=False):
		#print(f"Initializing With {self.refscan}\n")
		self.particles = self.motionmodel.init_particles()
		self.particles_init = self.particles.copy()
		self.ref_index = np.linspace(0,len(self.particles_init)-1,len(self.particles_init)).astype(int)
		self.ref_xy = self.motionmodel.xy
		self.reference_points = self.query_scan(self.refscan,self.motionmodel.xy)
		l = 0.1
		sigma2 = 0.01 #Observational Variance
		self.ref_mean = self.reference_points.mean(axis=0)
		self.ref_std = self.reference_points.std(axis=0)
		self.x_train = (self.reference_points - self.ref_mean)/self.ref_std
		K = np.exp(-dist(self.x_train[:,:2],self.x_train[:,:2],squared=True)/l**2) + sigma2*np.eye(self.x_train.shape[0])
		self.Kinv = np.linalg.inv(K)
		self.n = self.particles.shape[0]
		self.weights = np.ones(self.n)/self.n

	def particle_mean(self)-> np.ndarray:
		"Weighted Particle Mean [x, y, z, vx, vy, vz]"
		return np.average(self.particles, weights=self.weights, axis=0)		
		

		'''

		lamb = mean_test[:,None,:] - mean_ref[None,:,:]
		sigma = np.sum(cov_test[:,None,:] + cov_ref[None,:,:],axis=-1)


'''





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
		query_clouds = [self.query_scan(scan,self.particles[i,:2]-delta_0[i,:]) - delta_p[i] for i in range(self.particles.shape[0])]
		query_lls = np.zeros((len(query_clouds)))
		test_size = min(min(len(q) for q in query_clouds),150)

		for j,query_points in enumerate(query_clouds):
			rand_ind = np.random.choice(range(len(query_points)),test_size,replace=False)
			x_test = (query_points[rand_ind] - self.ref_mean)/self.ref_std
			Kstar = np.exp(-dist(x_test[:,:2],self.x_train[:,:2],squared=True)/l**2)
			Kss = np.exp(-dist(x_test[:,:2],x_test[:,:2],squared=True)/l**2)

			mu = Kstar @ self.Kinv @ self.x_train[:,2]
			Sigma = Kss - Kstar @ self.Kinv @ Kstar.T + sigma2*np.eye(Kss.shape[0])

			log_like = -0.5*(np.log(2*np.pi)*x_test.shape[0] + np.linalg.slogdet(Sigma)[1] + 0.5*(x_test[:,2]-mu)@np.linalg.inv(Sigma)@(x_test[:,2]-mu))
			query_lls[j] = log_like

		w = np.exp(0.2*(query_lls-query_lls.max())) + 1e-300
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