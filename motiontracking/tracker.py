import numpy as np 
import datetime
import sys
import matplotlib.pyplot as plt
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
		if self.refscan is not None:
			self.datetime = self.refscan.datetime
		self.calibrate = calibrate
		self.particles = None
		self.weights = None
		self.refclusters = None

		#self.reference_point = np.array([530884.00,7356524.00]) 
		self.initialize(self.calibrate)

	def query_scan(self,scan: Scan,loc):
		return scan.radialcluster(point=loc,radius=5)

	
	def getfeatures(self,particles,scan: Scan):
		features = [self.query_scan(scan,particles[i,:2]) for i in range(self.particles.shape[0])]
		return np.array(features)

	
	def initialize(self, calibrate=False):
		#print(f"Initializing With {self.refscan}\n")
		self.particles = self.motionmodel.init_particles()
		if not calibrate:
			self.reference_feature = self.query_scan(self.refscan,self.motionmodel.xy[:2])#self.gen_clusters(self.particles,self.refscan)
		self.n = self.particles.shape[0]
		self.weights = np.ones(self.n)/self.n

	def particle_mean(self)-> np.ndarray:
		"Weighted Particle Mean [x, y, z, vx, vy, vz]"
		return np.average(self.particles, weights=self.weights, axis=0)		
		

		'''

	def kernel_function(self,test,ref):
		mean_test = test.points
		mean_ref = ref.points

		cov_test = test.cov
		cov_ref = ref.cov


		lamb = mean_test[:,None,:] - mean_ref[None,:,:]
		sigma = np.sum(cov_test[:,None,:] + cov_ref[None,:,:],axis=-1)


		#detsig = np.linalg.det(sigma)
		sigma = np.tril(sigma).flatten()
		sigma = 1/sigma
		lamb = np.tril(lamb)
		lamb = np.sum(lamb.reshape(-1,lamb.shape[-1])**2,axis=-1)

		lengthscale = 0.5
		A = np.exp((-.5)*lamb*sigma)
		A /= 1/((2*np.pi**(3/2))*lengthscale**.5)
		A = np.sum(A)
		A/= mean_test.shape[0]*mean_ref.shape[0]

		return A
'''





	def scans_likelihood(self, scan: Scan,dt: datetime.timedelta=None):
		if dt is not None:
			self.dt = dt
		else:
			self.dt = scan.datetime - self.datetime


	
		self.datetime = scan.datetime
		self.refscan = scan

		self.particles = self.motionmodel.evolve_particles(self.particles,self.dt)
		testfeatures = self.getfeatures(self.particles,scan)
		log_likelihood = np.linalg.norm((testfeatures-self.reference_feature),axis=-1)

		return np.array(log_likelihood)


	
	def update_weights(self,scan: Scan,dt: datetime.timedelta=None):
		self.weights = np.exp(-self.scans_likelihood(scan, dt)) + 1e-300
		self.weights[~np.isnan(self.weights)] *= 1/self.weights[~np.isnan(self.weights)].sum()
		self.weights[np.isnan(self.weights)] = 0


	def systematic_resample(self):

		positions = (np.arange(self.n) + np.random.random() ) * (1/(self.n))
		cumulative_weight = np.cumsum(self.weights)
		cumulative_weight[-1] = 1
		indexes = np.searchsorted(cumulative_weight,positions)
		self.particles = self.particles[indexes,:]
		self.weights = self.weights[indexes]
		self.weights *= 1/self.weights.sum()


		posterior = self.particle_mean()
		self.reference_feature = self.query_scan(self.refscan,posterior[:2])
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
		#post = np.unique(indexes).shape[0] - indexes.shape[0]
		#print(f"\n Downsampled by {post} Weights\n")
		self.particles = self.particles[indexes,:]
		self.weights = self.weights[indexes]
		self.weights *= 1/self.weights.sum()
		posterior = self.particle_mean()
		self.reference_feature = self.query_scan(self.refscan,posterior[:2])
		return posterior

	


	def track(self, scan: Scan=None,calibrate: bool = False,dt: datetime.timedelta =None):
		if calibrate:
			#print(f"\n Calibrating From {self.refscan}\n")
			self.particles = self.motionmodel.evolve_particles(self.particles,dt)
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

		else:
			self.update_weights(scan)
			posterior = self.systematic_resample()
			#self.rewind()
			return [posterior,self.particles]