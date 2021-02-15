import numpy as np
import laspy as lp
import numpy as np
import scipy
from laspy.file import File
from scipy.spatial.ckdtree import cKDTree as KDTree
import time as time
import scipy.stats
from numpy.random import (normal,uniform)
import matplotlib.cm as cm
import itertools
from matplotlib.patches import Circle
from glob import glob
from os.path import join, splitext,basename
import rasterio
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from shapely.geometry import box 
import json
from scipy.interpolate import griddata
from rasterio import mask
import geopandas as gpd
from progress.bar import Bar
from numba import jit,njit,vectorize,jitclass,int32, float32

from sys import getsizeof
import datetime
import multiprocessing
from joblib import Parallel, delayed

from itertools import islice

def split_every(iterable,n):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))

class scan():
	def __init__(self,filepath):
		#start = time.time()
		self.name = filepath
		self.file = File(filepath,mode="r")
		#self.filesize = getsizeof(self.file)/8
		self.scale = self.file.header.scale[0]
		self.offset = self.file.header.offset[0]
		self.tree = KDTree(np.vstack([self.file.x, self.file.y, self.file.z]).transpose())
		#self.tree.size = getsizeof(self.tree)/8
		filename =  splitext( basename(filepath) )[0].replace("_","")
		dateobj = [int(filename[i:i+2]) for i in range(0,len(filename),2)] # year, month,day,hour,min,sec

		self.time = datetime.datetime(dateobj[0],dateobj[1],dateobj[2],dateobj[3],dateobj[4],dateobj[5],0)
		#print("File Size: {}, KDTree Size: {}\n".format(self.filesize,self.treesize))
		self.file=None
		#end = time.time() - start
		#print("Time Elapsed: {} for {}".format(int(np.rint(end)),basename(self.name)))

	def NNN(self,point,k):
		return self.tree.data[self.tree.query(point,k=k)[1]]

	def radialcluster(self,point,radius):
		neighbor = self.tree.data[self.tree.query(point,k=1)[1]]
		points = self.tree.data[self.tree.query_ball_point(neighbor,radius)]
		return np.array(points)


class Particleset():  
	def __init__(self,points=None,center=None,weight=None):
		self.points = points
		self.center = center
		self.weight = weight
		self.normalize()
		self.gen_covariance()

	def normalize(self):
		if self.center is not None:
			self.points[:,:2] -= np.mean(self.points[:,:2]) # normalize by either center or mean (TBD)

	def set_weight(self,weight):
		self.weight = weight

	def gen_covariance(self):
		if self.points is not None:
			self.cov = np.cov(self.points.T)

class Observer(Particleset):
    # should probably use kwargs here
	def __init__(self,initloc,std,initscan,N=5000):

		self.N = N
		initloc[-1]= initscan.NNN(initloc,k=1)[-1] # set initial elevation to NNN point elev. 
		self.initialize(initloc,std,N)
		self.refclusters = self.gen_clusters(initscan)
		self.time = initscan.time
		self.posteriors = [(initloc,std)]
		self.prior =  (np.array([1,.8,.6]),np.array([.3,.3,.2])) # mean and variance


        
	def initialize(self,initloc,std,N):
		self.particles = np.empty((N, 5))
		self.particles[:, 0] = normal(initloc[0],std[0],N) # x pos
		self.particles[:, 1] = normal(initloc[1],std[1],N) # y pos
		self.particles[:, 2] = normal(initloc[2],std[2],N) # elevation
		self.particles[:,3] = uniform(0,N,N)
		self.particles[:,4] = np.linspace(0,N,N).T # place holder

    
	def update(self):
		self.particles[:,3] /= np.sum(self.particles[:,3])
		cumsum = np.cumsum(self.particles[:,3])
		cumsum[-1] = 1
		indexes = np.searchsorted(cumsum,uniform(0,1,int(self.N))).astype(np.int)
		for i in range(len(indexes)):
		    self.particles[i,:] = self.particles[indexes[i],:]
		return indexes



	def evolve(self,dt):
		mean = dt*self.prior[0]
		var = dt*self.prior[1]
		update = np.zeros_like(self.particles)
		update[:,0] = normal(mean[0],var[0],self.N) #dynamical model for dx/dt
		update[:,1]  = normal(mean[1],var[1],self.N) #dynamical model for dy/dt
		update[:,2]  = normal(mean[2],var[2],self.N) #dynamical model for dz/dt
		self.particles += update


	def posterior(self,dt):

		mean = np.average(self.particles[:,:3],weights=self.particles[:,3],axis=0) # likelihood mean
		var = np.average((self.particles[:,:3]-mean)**2,weights=self.particles[:,3],axis=0) #likelihood var

		self.posteriors.append((mean,var))


	def query_scan(self,scan,loc):
		points = scan.NNN(point=loc,k=150)
		return Particleset(points=points,center=loc)
    

	def gen_clusters(self,scan):
		refclusters = []
		for center in list(self.particles[:,:3]) :
			refclusters.append(self.query_scan(scan,center))
		return refclusters


	def kernel_function(self,test,ref):
		# Evaluate the liklihood of one KDE under another
		# ** See https://iis.uibk.ac.at/public/papers/Xiong-2013-3DV.pdf [Section 3.3] For Details **
		mean_test = test.points
		cov_test = test.cov
		mean_ref = ref.points
		cov_ref = ref.cov
		assert mean_test.shape == mean_ref.shape
		gamma = mean_test - mean_ref
		sigma = cov_test + cov_ref
		A = 1/((2*np.pi**(3/2))*(np.linalg.det(sigma)**(.5)))
		B = np.exp((-.5)*gamma@np.linalg.inv(sigma)@gamma.T)
		C = 1/(mean_test.shape[0])
		return (C**2)*np.sum(A*B)
    
	def compare_scans(self,compare):
		#print("\nReferenece Time: {}, Compare Time {}\n".format(self.time,compare.time))
		dt = (compare.time - self.time) # compute the time diference between scans
		#print("Time Step: {}\n".format(dt.seconds))
		dt = dt.seconds/(3600*24)
		#print("Time Step: {} Days\n".format(dt))
		self.time = compare.time
		self.evolve(dt)
		test = self.gen_clusters(compare) # Produce a set of "test" locations for particles
		for i in range(self.N):
			self.particles[i,3] = self.kernel_function(test[i],self.refclusters[i]) # weight "Test locations"
		indexes = self.update()
		self.refclusters = np.array(test)[indexes] # resample the cluster set 
		self.posterior(dt)
		return dt
  
def bound_box(pointa,pointb):
	minx,miny = np.minimum(pointa[0],pointb[0]), np.minimum(pointa[1],pointb[1])
	maxx,maxy = np.maximum(pointa[0],pointb[0]), np.maximum(pointa[1],pointb[1])
	bbox = box(minx,miny,maxx,maxy)
	geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0],crs=32624) #24N
	return [json.loads(geo.to_json())['features'][0]['geometry']]



def obs_iteration(observer,currentscan,dx):
	observer.compare_scans(currentscan)
	displace = np.linalg.norm(observer.posteriors[-2][0]-observer.posteriors[-1][0])			
	mapping = observer.posteriors[-1][0].tolist()
	mapping.append(displace)
	dx.append(mapping)

if __name__ == "__main__":
	raster_path = "/home/dunbar/Research/helheim/data/2016_cpd_vels"
	lazfile_path = "/home/dunbar/Research/helheim/data/lazfiles"
	

	velorasters = sorted(velorasters, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
	lazfiles = glob(join(lazfile_path,"*.laz"))
	lazfiles = sorted(lazfiles, key=lambda i: int(os.path.splitext(os.path.basename(i))[0].strip("_") ))



	diagupp, diaglow = (535000.00,7359250.00),(537000.00,7358250.00)
	boundpolygon = bound_box(diagupp,diaglow)
	cpdvels = []
	for raster in velorasters:
		with rasterio.open(raster) as src:
			outimage = np.squeeze(mask.mask(src,boundpolygon,crop=True,filled=False)[0]) #extract mask
			cpdvels.append(np.squeeze(outimage))
	eastdims = (np.minimum(diagupp[0],diaglow[0]),np.maximum(diagupp[0],diaglow[0]),cpdvels[0].shape[1])
	northdims = (np.minimum(diagupp[1],diaglow[1]),np.maximum(diagupp[1],diaglow[1]),cpdvels[0].shape[0])


	easting = np.linspace(eastdims[0],eastdims[1],int(eastdims[2]/15))
	northing = np.linspace(northdims[0],northdims[1],int(northdims[2]/15))
	grid = [np.array([x,y,1000]) for x in easting for y in northing]
	posstd = np.array([.3,.3,.2])

	with ThreadPoolExecutor(8) as ex:
		initscan = ex.submit(scan,lazfiles[0]).result()
		observergrid = list(ex.map(lambda point: Observer(point,posstd,initscan), grid))


	#with Bar("Point Wise Grid Observation",fill='#', suffix='%(percent).1f%% - %(eta)ds') as bar:
	dxdt = []
	for i in range(1,len(lazfiles)):
		dx = []
		start = time.time()
		with ThreadPoolExecutor(8) as executor:
			currentscan= executor.submit(scan,lazfiles[i]).result()
			executor.map(lambda observer: obs_iteration(observer,currentscan,dx), observergrid)
		end = time.time() - start
		print("\n Time Per Scan: {} Minutes, Time per Obsever {} Seconds\n".format(end/60,end/len(grid)))
		#bar.next()
		dxdt.append(np.array(dx))
		currentscan = None
	dxdt = np.array(dxdt)


	#observergrid = None

	print("Begining Interpolation\n")
	easting = np.linspace(eastdims[0],eastdims[1],int(eastdims[2]))
	northing = np.linspace(northdims[0],northdims[1],int(northdims[2]))
	grideast,gridnorth = np.meshgrid(easting,northing)
	images = []
	for i in range(1,dxdt.shape[0]):
		try:
			image = griddata(points=dxdt[i,:,:2], values=dxdt[i,:,3][:,np.newaxis], xi=(grideast,gridnorth), method='linear')
		except:
			image = None
		images.append(np.squeeze(image))
	scan_displace = zip(lazfiles[1:],images)
	for lazpath, image in scan_displace:
		fname = splitext(lazpath)[0]
		plt.imshow(image)
		#plt.xticks(easting)
		#plt.yticks(northing)
		plt.colorbar()
		plt.xlabel('Easting')
		plt.ylabel('Northing')
		plt.title("Surface Displacement (M/D)")
		plt.savefig(fname)
		print("Saving: {} Interpolation\n".format(fname))
		hypothraster = os.path.join(raster_path,os.path.split(os.path.join(fname,"*.tif"))[1])
		if os.path.isfile(hypothraster):
			validation = cpdvels[velorasters.index(hypothraster)]
			fname = hypothraster.replace("*.tif","_cpdvels.tif")
			plt.imshow(validation)
			plt.colorbar()
			plt.savefig(fname)
			plt.clf()
