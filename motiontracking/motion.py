import cupy as np
import numpy
import datetime
from numpy.random import normal
from .raster import Raster


class CartesianMotion:



	def __init__(self,
		timestep: datetime.timedelta,
		xy: float,
		xy_sigma: float,
		vxyz: float,
		vxyz_sigma: float,
		axyz: float,
		axyz_sigma: float,
		DEM,
		n: int = 2000):

		self.n = n
		self.time_unit = timestep
		self.xy = xy
		self.xy_sigma = xy_sigma
		self.vxyz = vxyz
		self.vxyz_sigma = vxyz_sigma
		self.axyz = axyz
		self.axyz_sigma = axyz_sigma
		self.DEM = DEM
		

	def init_particles(self):
		"""
		Initialize particles around an initial mean position.
		Returns:
		    particles: Particle positions and velocities (x, y, z, vx, vy, vz).
		"""
		particles = np.zeros((self.n, 6), dtype=float)
		particles[:, 0:2] = np.array(self.xy) + np.array(self.xy_sigma)*np.random.randn(self.n,2)
		elevs = np.array(self.DEM.getval(particles[:,0:2].get()))
		particles[:,2] = elevs
		particles[:, 3:6] = np.array(self.vxyz) +np.array(self.vxyz_sigma)*np.random.randn(self.n,3)
		return particles


	def evolve_particles(self, particles: np.ndarray, dt: datetime.timedelta):
		"""
		Evolve particles through time by stochastic differentiation.
		Arguments:
		    particles: Particle positions and velocities (x, y, z, vx, vy, vz).
		    dt: Time step to evolve particles forward or backward.
		"""
		n = len(particles)
		time_units = dt.total_seconds() / self.time_unit.total_seconds()
		#print(f"\n Filter Time Units: {time_units}\n")
		
		axyz =  self.axyz + np.array(self.axyz_sigma)*np.random.randn(self.n,3)
		particles[:, 0:3] += (
		    time_units * particles[:, 3:6] + 0.5 * (axyz * time_units) ** 2
		)
		particles[:, 3:6] += time_units * axyz
	
	def devolve_particles(self, particles: np.ndarray, dt: datetime.timedelta,velocities):
		time_units = dt.total_seconds() / self.time_unit.total_seconds()
		particles[:, 0:3] -= (time_units *velocities)