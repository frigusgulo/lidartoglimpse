import numpy as np
import laspy as lp
import scipy
from laspy.file import File
from scipy.spatial import ConvexHull

from sklearn.neighbors import KDTree 
import pickle
from os.path import splitext,isfile
import time 
import datetime
from .filepath import Filepath
import matplotlib.pyplot as plt
from numpy.random import choice

class Scan():
	def __init__(self,filepath: Filepath = None,build_tree=True):

		if filepath is not None:
			self.filepath = filepath
			self.file = File(self.filepath.filepath,mode="r")
			self.header = self.file.header
			self.points = np.vstack([self.file.x, self.file.y, self.file.z]).transpose()
			self.tree = KDTree(self.points[:,:2],leaf_size=2**2,metric='euclidean')
			self.datetime = filepath.datetime

	def __getitem__(self,indices):
		return self.points[indices]

	def query(self,point,radius,maxpoints=1000,calibrate=False):
		point = np.array(point).flatten().reshape(1,2)
		#return self.tree.query(point,return_distance=False,k=25)[0]
		pointset =  self.tree.query_radius(point,r=radius)[0]
	
		if calibrate or len(pointset) >= maxpoints/6 :
			return pointset

		else:
			return self.tree.query(point,return_distance=False,k=int(maxpoints//5))[0]
			


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

	def convexhull(self,vertices=True):
		if vertices:
			return ConvexHull(self.points).vertices
		else:
			return ConvexHull(self.points)

		# pass to glimpse.raster.rasterize_polygons()

