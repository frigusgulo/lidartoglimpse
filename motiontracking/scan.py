import numpy as np
import laspy as lp
import scipy
from laspy.file import File

from sklearn.neighbors import KDTree 
import pickle
from os.path import splitext,isfile
import time 
import datetime
from .filepath import Filepath
import matplotlib.pyplot as plt

class Scan():
	def __init__(self,filepath: Filepath = None):

		if filepath is not None:
			self.filepath = filepath
			self.file = File(self.filepath.filepath,mode="r")
			self.header = self.file.header
			self.points = np.vstack([self.file.x, self.file.y, self.file.z]).transpose()
			self.tree = KDTree(self.points[:,:2],leaf_size=2**2,metric='euclidean')
			self.datetime = filepath.datetime


	def __getstate__(self):
		return self.__dict__

	def __setstate__(self,data):
		self.__dict__ = data

	def radialcluster(self,point,radius=15):

		point = np.array(point).flatten().reshape(1,2)
	
		# Return a descriptive vector of the radially queried surface points from the scan
		dist,ind = self.tree.query(point)
		point_nn = self.points[int(ind),:2].reshape(1,-1)
		points =  self.tree.query_radius(point_nn,r=radius)[0]
		points = points.astype(np.int)
		minimum = min(1000,len(points))
		np.random.shuffle(points)
		points = points[:minimum]
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

