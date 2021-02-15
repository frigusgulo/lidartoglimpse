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
#import motiontracking

from motiontracking.motion import CartesianMotion
from motiontracking.scanset import Scanset
from motiontracking.raster import Raster
from motiontracking.filepath import Filepath
from motiontracking.scan import Scan, Particleset
from motiontracking.scanset import Scanset
from motiontracking.tracker import Tracker

'''
Compare particle filter estimates with cpd vels via sampling locations from the rasters and plotting them on a line.
'''




raster_path = "/home/dunbar/Research/helheim/data/2016_cpd_vels"
lazfile_path = "data/downsampledlazfiles/"
tracks = np.load("data/tracks.npy",allow_pickle=True) # file , point, position
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



#print(tracks.shape)
velocities = np.sqrt(tracks[:,:,3]**2 +  tracks[:,:,4]**2 )# + tracks[:,5,:]**2)
tracks = np.delete(tracks,np.s_[3:6],2)
print(tracks.shape)
tracks[:,:,-1] = velocities
print(tracks[0,:,:])
#tracks = tracks[:,:,::-1]


scan_cpd_pairs = []
for i, (cpd,preds) in enumerate(mapping):

	try:
		# Locate the closest match between the lazfiles and cpd files
		#print(f"\n{basename(cpd)} Found For Laz file {basename(preds)}\n")
		scanpreds = tracks[i,:,:]
		scan_cpd_pairs.append((cpd,scanpreds,preds))
	except:
		pass

for rasterpath,scanpreds,lazfile in scan_cpd_pairs:
	print(f"\nOpening {basename(rasterpath)} for: \n {lazfile}\n")
	raster = Raster(rasterpath)
	#mask  = ~np.isnan(scanpreds[:,-1])
	#scanpreds = scanpreds[mask,:]

	print(scanpreds)
	cpdvals = 24*np.array(raster.getval(scanpreds[:,:2],1))
	preds = scanpreds[:,2][:,np.newaxis]


	print(preds.shape)
	print(cpdvals.shape)
	#cpdvals = cpdvals[::-1,:]


	slope,intercept,r_vl,p_vl,std_err = linregress(preds,cpdvals)

	print(f"Stats: \n\
		Slope: {slope}\n\
		Correlation: {r_vl}\n")



	locs = np.array([raster.dem.index(scanpreds[i,0],scanpreds[i,1]) for i in range(scanpreds.shape[0])])

	xlin = np.linspace(0,24)
	ylin = xlin*slope + intercept
	fig,(ax1,ax2) = plt.subplots(1,2)
	ax1.plot(xlin,ylin,'b',label='Line of Fit')
	ax1.plot(xlin,xlin,'g',label='Y=X')
	ax1.legend()
	ax1.scatter(preds,cpdvals,c='r')
	ax1.set_xlabel("Estimated Filter Values M/D")
	ax1.set_ylabel("Estimated CPD Values M/D")
	#ax1.set_xlim(0,1.5)
	#ax1.set_ylim(0,1.5)
	ax2.scatter(locs[:,1],locs[:,0],c='r')
	ax2.imshow(raster.dem.read(1))
	plt.show()
