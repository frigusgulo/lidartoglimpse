import numpy as np
import laspy as lp 
import os
import datetime
from glob import glob
from os.path import join
import json
import rasterio
from rasterio import mask
from multiprocessing import Pool 
from progress.bar import Bar
import warnings
from operator import methodcaller
import glimpse
import pdb
from motiontracking.motion import CartesianMotion
from motiontracking.scanset import Scanset
from motiontracking.raster import Raster
from motiontracking.filepath import Filepath
from motiontracking.scan import Scan
from motiontracking.scanset import Scanset
from motiontracking.tracker import Tracker

warnings.filterwarnings('ignore')
raster_path = "/home/dunbar/Research/helheim/data/2016_cpd_vels"
lazfile_path = "/home/dunbar/Research/helheim/data/lazfiles"
#lazfile_path = "/home/dunbar/Research/helheim/lidartoglimpse/data/downsampledlazfiles"
velorasters = glob(join(raster_path,"*.tif"))
lazfiles = glob(join(lazfile_path,"*.laz"))
lazfiles = [ Filepath(x) for x in lazfiles ]
lazfiles.sort(key= lambda i: i.datetime) 
cpdvels = [Filepath(x) for x in velorasters]
cpdvels.sort(key = lambda i: i.datetime)
cpdvelsimgs = []
diagupp, diaglow = (535800.00,7359300.00),(536300.00,7358300)

eastdims = (np.minimum(diagupp[0],diaglow[0]),np.maximum(diagupp[0],diaglow[0])) 
northdims = (np.minimum(diagupp[1],diaglow[1]),np.maximum(diagupp[1],diaglow[1]))


easting = np.linspace(eastdims[0],eastdims[1],8).tolist()
northing = np.linspace(northdims[0],northdims[1],8).tolist()
grid = [[x,y] for x in easting for y in northing][:8]

#grid = grid[::1]

#initialscan = Scan(lazfiles[0])


DEM = Raster("/home/dunbar/Research/helheim/data/observations/dem/helheim_wgs84UTM.tif")
motionparams = {
	"timestep": datetime.timedelta(days=1),
	"DEM": DEM,
	"xy_sigma": np.array([.1,.1]),
	"vxyz": np.array([10,-6,0]),
	"vxyz_sigma": np.array([6,6,0.1]),
	"axyz": np.array([0,0,0]),
	"axyz_sigma": np.array([3,3,0.02]),
	"n": 3500
}



DATA_DIR = "/home/dunbar/Research/helheim/data/observations"
observerpath = ['stardot2']
observers = []
for observer in observerpath:
    path = join(DATA_DIR,observer)
    campaths = glob(join(path,"*.JSON"))
    images = [glimpse.Image(path=campath.replace(".JSON",".jpg"),cam=glimpse.Camera.from_json(campath)) for campath in campaths]
    images.sort(key= lambda img: img.datetime)
    datetimes = np.array([img.datetime for img in images])
    for n, delta in enumerate(np.diff(datetimes)):
        if delta <= datetime.timedelta(seconds=0):
            secs = datetime.timedelta(seconds= n%5 + 1)
            images[n+1].datetime = images[n+1].datetime + secs
    diffs = np.array([dt.total_seconds() for dt in np.diff(np.array([img.datetime for img in images]))])
    negate = diffs[diffs <= 1].astype(np.int)
    [images.pop(_) for _ in negate]

   
images = np.array(images)
  
nearest_scan = np.searchsorted([laz.datetime for laz in lazfiles],images[0].datetime)
start = lazfiles[nearest_scan-5].datetime

scanset = Scanset(lazfiles,images=images)
stopdate = datetime.timedelta(days=45)
obs_inds = scanset.get_inds(start,lazfiles[0].datetime+stopdate)
dummy_img = "/home/dunbar/Research/helheim/data/observations/stardot2/HEL_DUAL_StarDot2_20200309_210000.jpg"
dummy_cam = "/home/dunbar/Research/helheim/data/observations/stardot2/HEL_DUAL_StarDot2_20200309_210000.JSON"
imageset = []
for i in range(obs_inds.shape[0]):
	testobs = scanset.observations[i]
	if isinstance(testobs,Filepath):
		imageset.append(glimpse.Image(path=dummy_img,cam=glimpse.Camera.from_json(dummy_cam),datetime=testobs.datetime))
	else:
		imageset.append(testobs)

stardot2 = glimpse.Observer(np.array(imageset[1:]))


DATA_DIR = "/home/dunbar/Research/helheim/data/observations"
DEM_DIR = os.path.join(DATA_DIR, 'dem')
path = glob(join(DEM_DIR,"*.tif"))[0]
print("DEM PATH: {}".format(path))
dem = glimpse.Raster.open(path=path)

dem.crop(zlim=(0, np.inf))
dem.fill_crevasses(mask=~np.isnan(dem.array), fill=True)
for obs in [stardot2]:
    dem.fill_circle(obs.images[0].cam.xyz, radius=50)
viewshed = dem.copy()
viewshed.array = np.ones(dem.shape, dtype=bool)
for obs in observers:
    viewshed.array &= dem.viewshed(obs.images[0].cam.xyz,correction=True)



initialscan = scanset.obs_index(obs_inds[0])




# CPD resolution is 15x15 meters per cell
trackergrid = [Tracker(CartesianMotion(xy=point,**motionparams),initialscan) for point in grid]
imagetrackergrid = [glimpse.Tracker(observers=[stardot2],viewshed=viewshed) for tracker in trackergrid]

#print(f"\n Observing At {len(trackergrid)} Points \n")
tile_size = (15,15)
MSE = []
MED_SE = []
tracks = []
particle_set = []
observations = []
a = len(lazfiles)-1
timecounter = 0
obs_type = []
#indexset = np.arange(1,o).tolist()
for i in obs_inds[1:].tolist():
	testobs = scanset.obs_index(i)
	if isinstance(testobs,Scan):
		#print(testobs.datetime)
		#pool.map(methodcaller('track',testscan),trackergrid)
		obs_type.append(0)
		[tracker.track(observation=testobs) for tracker in trackergrid]
		observations.append([testobs.datetime,0])
		MSE.append(np.sqrt(np.mean([tracker.error[-1] for tracker in trackergrid])))
		MED_SE.append(np.sqrt(np.median([tracker.error[-1] for tracker in trackergrid])))
		#print(f"Root Mean Square Error: {MSE[:]}\n")
		status = [[SE,obs] for (SE,obs) in zip(MED_SE[:],obs_type)]
		print(f"Root Median Square Error: {status}\n")
		#print(observations)
		timecounter += trackergrid[0].dt.total_seconds()/trackergrid[0].motionmodel.time_unit.total_seconds()
		if timecounter >=8 :
			[it.reset() for it in imagetrackergrid]
			[tracker.re_initialize(testobs) for tracker in trackergrid]
	elif isinstance(testobs,glimpse.Image):
	
		for j,(imagetracker,tracker) in enumerate(zip(imagetrackergrid,trackergrid)):
			#try:
			dt = testobs.datetime-tracker.datetime
			tracker.datetime = testobs.datetime
			obs_type.append(1)
			updated_particles = tracker.motionmodel.evolve_particles(tracker.particles,dt)
			tracker.particles = updated_particles
			imagetracker.particles = updated_particles
			imagetracker.test_particles()
			
			timecounter += dt.total_seconds()/trackergrid[0].motionmodel.time_unit.total_seconds()

			if imagetracker.templates is None:
				try:
					imagetracker.initialize_template(0,i,tile_size=tile_size)
					#print(f"Template Initialized")

				except Exception as e:
					pass
					#print(f" {j},{e}")
			elif imagetracker.templates[0] is not None :
				try:
				#pdb.set_trace()
					log_likes = imagetracker.compute_observer_log_likelihoods(0,i)
					#pdb.set_trace()
					likelihoods = np.exp((-1)*log_likes)
					likelihoods *= 1/likelihoods.sum()
					tracker.track(testobs,likelihoods=likelihoods)
					MSE.append(np.sqrt(np.mean([tracker.error[-1] for tracker in trackergrid])))
					MED_SE.append(np.sqrt(np.median([tracker.error[-1] for tracker in trackergrid])))
					observations.append([testobs.datetime,1])
					#print(f"Image {i} Used for {j}")
				except Exception as e:
					pass
					#print(f" {j}: {e} ")

		



#scandatetimes = matplotlib.dates.date2num([date for date in scanset.scans_dt])
tracks = np.array([tracker.posterior_set for tracker in trackergrid])
particles = np.array([tracker.particle_set for tracker in trackergrid])
covariances = np.array([tracker.covariance_set for tracker in trackergrid])
timesteps  = np.array([tracker.timestep_set for tracker in trackergrid])
error  = np.array([tracker.error for tracker in trackergrid])
radii_sigma2 = np.array([(tracker.radius,tracker.sigma2) for tracker in trackergrid])

np.save("data/observations_2",observations)
np.save("data/covariances_2",covariances)
np.save("data/tracks_2",tracks)
np.save("data/particles_2",particles)
np.save("data/timesteps_2",timesteps)
np.save("data/error_2",error)
np.save("data/MSE_2",MSE)
np.save("data/MED_SE_2",MED_SE)
np.save("data/radii_2",radii_sigma2)

# meanvector = tracker.track(scan)

