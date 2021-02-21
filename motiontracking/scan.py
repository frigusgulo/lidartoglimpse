import numpy as np
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
		self.points = np.vstack([self.file.x, self.file.y, self.file.z]).transpose()
		self.tree = KDTree(self.points[:,:2])
		
		
		self.datetime = filepath.datetime

	def radialcluster(self,point,radius):

		# Return a descriptive vector of the radially queried surface points from the scan
		point_nn = self.tree.data[self.tree.query(point,k=1)[1]]
		points =  self.tree.query_ball_point(point_nn,radius,n_jobs=-1)
		minimum = min(1000,len(points))
		np.random.shuffle(points)
		points = points[:minimum]
		points = [int(i) for i in points]
		return self.points[points]


	def downsample(self,filepath,skipinterval,xbounds,ybounds):
		Xvalid = np.logical_and((np.min(xbounds) <= self.file.x),np.max(xbounds) >= self.file.x)
		Yvalid = np.logical_and((np.min(ybounds) <= self.file.y),np.max(ybounds) >= self.file.y)
		keep = np.where(np.logical_and(Xvalid,Yvalid))
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