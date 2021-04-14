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
diagupp, diaglow = (535500.00,7358300.00),(536500.00,7359300)

eastdims = (np.minimum(diagupp[0],diaglow[0]),np.maximum(diagupp[0],diaglow[0])) 
northdims = (np.minimum(diagupp[1],diaglow[1]),np.maximum(diagupp[1],diaglow[1]))


easting = np.linspace(eastdims[0],eastdims[1],8).tolist()
northing = np.linspace(northdims[0],northdims[1],8).tolist()
grid = [[x,y] for x in easting for y in northing]#[:8]

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

    print("Image set {} \n".format(len(images)))
    obs = glimpse.Observer(list(np.array(images)),cache=False)
    observers.append(obs)

images = observers[0].images

scanset = Scanset(lazfiles,images=images)
stopdate = datetime.timedelta(days=45)
obs_inds = scanset.get_inds(lazfiles[0].datetime,lazfiles[0].datetime+stopdate)
initialscan = scanset.obs_index(obs_inds[0])
# CPD resolution is 15x15 meters per cell
trackergrid = [Tracker(CartesianMotion(xy=point,**motionparams),initialscan) for point in grid]

print(f"\n Observing At {len(trackergrid)} Points \n")

MSE = []
MED_SE = []
tracks = []
particle_set = []
observations = []
a = len(lazfiles)-1
#indexset = np.arange(1,o).tolist()
for i in obs_inds[1:].tolist():
	testobs = scanset.obs_index(i)
	if isinstance(testobs,Scan):
		#print(testobs.datetime)
		#pool.map(methodcaller('track',testscan),trackergrid)

		[tracker.track(observation=testobs) for tracker in trackergrid]
		observations.append([testobs.datetime,0])
		MSE.append(np.sqrt(np.mean([tracker.error[-1] for tracker in trackergrid])))
		MED_SE.append(np.sqrt(np.median([tracker.error[-1] for tracker in trackergrid])))
		print(f"Root Mean Square Error: {MSE[:]}\n")
		print(f"Root Median Square Error: {MED_SE[:]}\n")

#scandatetimes = matplotlib.dates.date2num([date for date in scanset.scans_dt])
tracks = np.array([tracker.posterior_set for tracker in trackergrid])
particles = np.array([tracker.particle_set for tracker in trackergrid])
covariances = np.array([tracker.covariance_set for tracker in trackergrid])
timesteps  = np.array([tracker.timestep_set for tracker in trackergrid])
error  = np.array([tracker.error for tracker in trackergrid])
radii_sigma2 = np.array([(tracker.radius,tracker.sigma2) for tracker in trackergrid])

np.save("data/observations",observations)
np.save("data/covariances",covariances)
np.save("data/tracks",tracks)
np.save("data/particles",particles)
np.save("data/timesteps",timesteps)
np.save("data/error",error)
np.save("data/MSE",MSE)
np.save("data/MED_SE",MED_SE)
np.save("data/radii",radii_sigma2)

# meanvector = tracker.track(scan)