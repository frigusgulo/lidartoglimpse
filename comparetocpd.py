import numpy as np
import laspy as lp 
import os
import datetime
from glob import glob
from os.path import join,basename,splitext
from shapely.geometry import box 
import geopandas as gpd
import json
import rasterio
from rasterio import mask
from concurrent.futures import ThreadPoolExecutor
from progress.bar import Bar
import matplotlib.pyplot as plt
from scipy.stats.mstats import linregress
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import motiontracking

from motiontracking.motion import CartesianMotion
from motiontracking.scanset import Scanset
from motiontracking.raster import Raster
from motiontracking.filepath import Filepath
from motiontracking.scan import Scan
from motiontracking.scanset import Scanset
from motiontracking.tracker import Tracker

'''
Compare particle filter estimates with cpd vels via sampling locations from the rasters and plotting them on a line.
'''




raster_path = "/home/dunbar/Research/helheim/data/2016_cpd_vels"
lazfile_path = "/home/dunbar/Research/helheim/downsampledlazfiles"
tracks = np.load("data/tracks.npy",allow_pickle=True) # file , point, position

covars = np.load("data/covariances.npy",allow_pickle=True)

if(len(tracks.shape)==2):
	tracks = tracks[np.newaxis,:,:]


velorasters = glob(join(raster_path,"*.tif"))
print(f"\n Found {len(velorasters)} Raster Files\n")
lazfiles = glob(join(lazfile_path,"*.dslaz"))
print(f"\n Found {len(lazfiles)} Laz Files\n")



lazfiles = [ Filepath(x) for x in lazfiles ]
lazfiles.sort(key= lambda i: i.datetime) 
cpdvels = [Filepath(x) for x in velorasters]
cpdvels.sort(key = lambda i: i.datetime)
#print(len(cpdvels))

lazfiles_strip = [splitext(basename(file.filepath))[0].split("_")[0] for file in lazfiles]
cpdvels_strip = [splitext(basename(file.filepath))[0].split("_")[0] for file in cpdvels]

indexes = np.searchsorted(cpdvels_strip,lazfiles_strip)
#print("\n\n Indexes: ",indexes)
#print(np.array(cpdvels_strip)[indexes])

cpdvels = np.array(cpdvels)[indexes].tolist()
mapping = [(cpd.filepath,laz.filepath) for (cpd,laz) in zip(cpdvels,lazfiles)]

vx_var = np.squeeze(np.sqrt(covars[:,3,3,:]))
vy_var = np.squeeze(np.sqrt(covars[:,4,4,:]))


velocities = np.sqrt(tracks[:,:,3]**2 +  tracks[:,:,4]**2 )# + tracks[:,5,:]**2)
tracks = np.delete(tracks,np.s_[3:6],2)

tracks[:,:,-1] = velocities

scan_cpd_pairs = []
for i, (cpd,preds) in enumerate(mapping):

	try:
		# Locate the closest match between the lazfiles and cpd files
		#print(f"\n{basename(cpd)} Found For Laz file {basename(preds)}\n")
		scanpreds = tracks[i,:,:]
		scan_cpd_pairs.append((cpd,scanpreds,preds))
	except:
		pass

for i,(rasterpath,scanpreds,lazfile) in enumerate(scan_cpd_pairs):
	print(f"\nOpening {basename(rasterpath)} for: \n {lazfile}\n")
	#raster = Raster("/home/dunbar/Downloads/TSX_E66.50N_07Aug16_18Aug16_09-23-32_vv_v03.0.tif")

	raster = Raster(rasterpath)
	scanpreds = scanpreds[~np.isnan(scanpreds)]
	
	scanpreds = np.reshape(scanpreds,(scanpreds.shape[0]//3,3))
	locs = [raster.dem.index(xy[0],xy[1]) for xy in scanpreds[:,:2].tolist()]
	cpdvals = (24)*np.array([raster.dem.read(1)[loc] for loc in locs])

	
	
	
	preds = scanpreds[:,2][:,np.newaxis]


	'''
	slope,intercept,r_vl,p_vl,std_err = linregress(preds,cpdvals)

	print(f"Stats: \n\
		Slope: {slope}\n\
		Correlation: {r_vl}\n")
	'''
	#scanpreds[:,:2] -=scanpreds[0,:2]
	scanpreds[:,:2] = scanpreds[:,:2].astype(int)
	colors = cm.rainbow(np.linspace(0,1,preds.shape[0]))


	locs = np.array(locs)
	xlin = np.linspace(0,24)
	#ylin = xlin*slope + intercept
	fig,(ax1,ax2) = plt.subplots(1,2)
	#ax1.plot(xlin,ylin,'b',label='Line of Fit')
	ax1.plot(xlin,xlin,'g',label='Y=X')
	ax1.legend()
	ax1.scatter(preds,cpdvals,color=colors)
	ax1.errorbar(preds,cpdvals,xerr=vx_var,yerr=vy_var)
	ax1.set_xlabel("Estimated Filter Values M/D")
	ax1.set_ylabel("Estimated CPD Values M/D")


	#ax1.set_xlim(0,1.5)
	#ax1.set_ylim(0,1.5)
	ax2.scatter(locs[:,1],locs[:,0],color=colors)
	ax2.imshow(raster.dem.read(1))
	plt.show()
