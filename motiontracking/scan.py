import cupy as np
import numpy
import laspy as lp
import scipy
from laspy.file import File
from scipy.spatial.ckdtree import cKDTree as KDTree
import time 
import datetime
from .filepath import Filepath
import matplotlib.pyplot as plt

class Scan():
	def __init__(self,filepath: Filepath):

		self.filepath = filepath
		self.file = File(self.filepath.filepath,mode="r")
		self.header = self.file.header
		self.points = numpy.vstack([self.file.x, self.file.y, self.file.z]).transpose()
		self.tree = KDTree(self.points[:,:2])
		
		
		self.datetime = filepath.datetime

		#print(f"Instantiating Scan From {self.filepath.filepath}\n")
	def knn(self,point,k):
		point = np.asnumpy(point)
		return self.points[self.tree.query(point,k=k)[1],:]

	def radialcluster(self,point,radius):

		# Return a descriptive vector of the radially queried surface points from the scan
		point_nn = self.tree.query(point,k=1)[1]
		point_nn = np.array(self.points[point_nn,:])[:2]

		locs = np.array(self.points[self.tree.query_ball_point(point_nn.get(),radius,n_jobs=-1),:])
	
		distances = np.linalg.norm((locs[:,:2] - point_nn),axis=-1)
		weights = distances/distances.sum()
		geometricmean = np.average(locs,weights=weights,axis=0)
		locs -= geometricmean
		cov = (1/locs.shape[0])*locs.T@locs
		eigs = np.linalg.eigvalsh(cov)
		feats = np.array([eigs[0]-eigs[1],eigs[1]-eigs[2],eigs[2]])/eigs[0]
		descriptor = np.hstack((eigs.flatten(),feats.flatten()))
		return descriptor
	

	def downsample(self,filepath,skipinterval,xbounds,ybounds):
		Xvalid = numpy.logical_and((numpy.min(xbounds) <= self.file.x),numpy.max(xbounds) >= self.file.x)
		Yvalid = numpy.logical_and((numpy.min(ybounds) <= self.file.y),numpy.max(ybounds) >= self.file.y)
		keep = numpy.where(numpy.logical_and(Xvalid,Yvalid))
		points = self.points[keep]
		shuffleinds = np.random.shuffle(np.arange(points.shape[0]))
		points = np.array(points)[shuffleinds]
		points = np.squeeze(points[::skipinterval].transpose())
		with File(filepath,mode='w',header=self.header) as output:
			output.points = points

	def plot(self,points=None):
		plt.scatter(self.points[:,0],self.points[:,1],c='r',s=3)
		if points is not None:
			plt.scatter(points[:,0].get(),points[:,1].get(),c='g',s=30)
		plt.show()



class Particleset():  
	def __init__(self,points=None,distances=None):
		self.points = points
		self.distances = distances
		self.normalize()

	def normalize(self):
		weights = self.distances/self.distances.sum()
		geometricmean = np.average(self.points,weight=weights)
		self.points -= geometricmean
		cov = (1/self.points.shape[0])*self.points@self.points.T
		self.eigs = np.linalg.eig(cov,homogeneous_eigvals=True)
		self.feats = np.array([self.eigs[0]-self.eigs[1],self.eigs[1]-self.eigs[2],self.eigs[2]])/self.eigs[0]
		self.descriptor = np.hstack((self.eigs,self.feats))