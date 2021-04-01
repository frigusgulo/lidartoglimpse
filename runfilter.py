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
diagupp, diaglow = (535500.00,7358300.00),(536500.00,7359000)

eastdims = (np.minimum(diagupp[0],diaglow[0]),np.maximum(diagupp[0],diaglow[0])) 
northdims = (np.minimum(diagupp[1],diaglow[1]),np.maximum(diagupp[1],diaglow[1]))


easting = np.linspace(eastdims[0],eastdims[1],8).tolist()
northing = np.linspace(northdims[0],northdims[1],8).tolist()
grid = [[x,y] for x in easting for y in northing]#[:1]

#grid = grid[::1]

#initialscan = Scan(lazfiles[0])
scanset = Scanset(lazfiles)
initialscan = scanset.index(0)
# CPD resolution is 15x15 meters per cell

DEM = Raster("/home/dunbar/Research/helheim/data/observations/dem/helheim_wgs84UTM.tif")
motionparams = {
	"timestep": datetime.timedelta(days=1),
	"DEM": DEM,
	"xy_sigma": np.array([.1,.1]),
	"vxyz": np.array([10,-5,0]),
	"vxyz_sigma": np.array([6,6,0.1]),
	"axyz": np.array([0,0,0]),
	"axyz_sigma": np.array([3,3,0.02]),
	"n": 2000
}

#with ThreadPoolExecutor(max_workers=12) as executor:
	#trackergrid = [executor.submit(Tracker(CartesianMotion(xy=point,**motionparams),initialscan)) for point in grid]

trackergrid = [Tracker(CartesianMotion(xy=point,**motionparams),initialscan) for point in grid]

print(f"\n Observing At {len(trackergrid)} Points \n")

#Un comment for calibration
'''
calibration_iters = 2**1
timestep = np.ones(calibration_iters)
timestep[::2]*=-1
timestep = timestep.tolist()

particle_set = []
for time in timestep:
	dt = datetime.timedelta(hours=12*time)
	[tracker.track(calibrate=True,dt=dt) for tracker in trackergrid]


#np.save("data/calibration_particles",particle_set)
print(f"\n\n Calibration Complete\n\n")
'''
#pool = Pool(processes=12)


tracks = []
particle_set = []
a = len(lazfiles)-1
indexset = np.arange(1,a).tolist()
for i in indexset:
	testscan = scanset.index(i)
	#pool.map(methodcaller('track',testscan),trackergrid)
	[tracker.track(testscan) for tracker in trackergrid]


tracks = np.array([tracker.posterior_set for tracker in trackergrid])
particles = np.array([tracker.particle_set for tracker in trackergrid])
covariances = np.array([tracker.covariance_set for tracker in trackergrid])
timesteps  = np.array([tracker.timestep_set for tracker in trackergrid])

np.save("data/covariances",covariances)
np.save("data/tracks",tracks)
np.save("data/particles",particles)
np.save("data/timesteps",timesteps)



# meanvector = tracker.track(scan)