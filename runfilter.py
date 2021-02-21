import numpy as np
import laspy as lp 
import os
import datetime
from glob import glob
from os.path import join
import json
import rasterio
from rasterio import mask
from concurrent.futures import ThreadPoolExecutor
from progress.bar import Bar


from motiontracking.motion import CartesianMotion
from motiontracking.scanset import Scanset
from motiontracking.raster import Raster
from motiontracking.filepath import Filepath
from motiontracking.scan import Scan, Particleset
from motiontracking.scanset import Scanset
from motiontracking.tracker import Tracker


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


easting = np.linspace(eastdims[0],eastdims[1],10).tolist()
northing = np.linspace(northdims[0],northdims[1],10).tolist()
grid = [[x,y] for x in easting for y in northing]

mid = len(grid)//2
grid = grid[:1]

initialscan = Scan(lazfiles[0])
scanset = Scanset(lazfiles[1:5])

# CPD resolution is 15x15 meters per cell

DEM = Raster("/home/dunbar/Research/helheim/data/observations/dem/helheim_wgs84UTM.tif")
motionparams = {
	"timestep": datetime.timedelta(days=1),
	"DEM": DEM,
	"xy_sigma": np.array([0.1,0.1]),
	"vxyz": np.array([0,0,0]),
	"vxyz_sigma": np.array([11,11,0.1]),
	"axyz": np.array([0,0,0]),
	"axyz_sigma": np.array([3,3,0.02])
}

#with ThreadPoolExecutor(max_workers=12) as executor:
	#trackergrid = [executor.submit(Tracker(CartesianMotion(xy=point,**motionparams),initialscan)) for point in grid]

trackergrid = [Tracker(CartesianMotion(xy=point,**motionparams),initialscan) for point in grid]
print(f"\n Observing At {len(trackergrid)} Points \n")



#Un comment for calibration

calibration_iters = 2**3
timestep = np.ones(calibration_iters)
timestep[0::2]*=-1
timestep = timestep.tolist()

particle_set = []
tracker = trackergrid[0]
for time in timestep:
	dt = datetime.timedelta(hours=time)
	tracker.track(calibrate=True,dt=dt)


#np.save("data/calibration_particles",particle_set)
print(f"\n\n Calibration Complete\n\n")



tracks = []
particle_set = []
for i in range(len(scanset.scans)):
	testscan = scanset.index(i)
	means,particles = zip(*[tracker.track(testscan) for tracker in trackergrid])
	testscan=None
	tracks.append(means[:])
	particle_set.append(particles[:])


tracks = np.array(tracks)
particles = np.array(particle_set)


np.save("data/tracks",tracks)
np.save("data/particles",particles)



# meanvector = tracker.track(scan)