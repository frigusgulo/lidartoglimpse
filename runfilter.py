import cupy as np
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
#lazfile_path = "/home/dunbar/Research/helheim/data/lazfiles"
lazfile_path = "/home/dunbar/Research/helheim/lidartoglimpse/data/downsampledlazfiles"
velorasters = glob(join(raster_path,"*.tif"))
lazfiles = glob(join(lazfile_path,"*.dslaz"))
lazfiles = [ Filepath(x) for x in lazfiles ]
lazfiles.sort(key= lambda i: i.datetime) 
cpdvels = [Filepath(x) for x in velorasters]
cpdvels.sort(key = lambda i: i.datetime)
cpdvelsimgs = []
diagupp, diaglow = (535500.00,7358300.00),(536300.00,7359300)

eastdims = (np.minimum(diagupp[0],diaglow[0]),np.maximum(diagupp[0],diaglow[0])) 
northdims = (np.minimum(diagupp[1],diaglow[1]),np.maximum(diagupp[1],diaglow[1]))


easting = np.linspace(eastdims[0],eastdims[1],10).tolist()
northing = np.linspace(northdims[0],northdims[1],10).tolist()
grid = [[x,y] for x in easting for y in northing]

mid = len(grid)//2
#grid = [grid[mid]]

initialscan = Scan(lazfiles[0])
scanset = Scanset(lazfiles[1:])

# CPD resolution is 15x15 meters per cell

DEM = Raster("/home/dunbar/Research/helheim/data/observations/dem/helheim_wgs84UTM.tif")
motionparams = {
	"timestep": datetime.timedelta(days=1),
	"DEM": DEM,
	"xy_sigma": np.array([5,5]),
	"vxyz": np.array([10,10,1]),
	"vxyz_sigma": np.array([4.5,4.5,2]),
	"axyz": np.array([0,0,0]),
	"axyz_sigma": np.array([1,1,.1])
}


trackergrid = [Tracker(CartesianMotion(xy=point,**motionparams),initialscan) for point in grid]
print(f"\n Observing At {len(trackergrid)} Points \n")


calibration_iters = 2**2
timestep = np.ones(calibration_iters)
timestep[::2]*=-1
timestep = timestep.tolist()
with ThreadPoolExecutor(max_workers=8) as executor:
	for time in timestep:
		dt = datetime.timedelta(hours=time)
		executor.map(lambda tracker: tracker.track(calibrate=True,dt=dt),trackergrid)
print(f"\n\n Calibration Complete\n\n")

#points = np.array([tracker.particle_mean() for tracker in trackergrid])
#trackergrid[0].refscan.plot(points[:,0:2])

#for tracker in trackergrid:
	#tracker.refscan.plot(points=tracker.particle_mean())

tracks = []
particle_set = []
for i in range(len(scanset.scans)):


	testscan = scanset.index(i)
	means,particles = zip(*[tracker.track(scan=testscan) for tracker in trackergrid])
	tracks.append(means[:])
	particle_set.append(particles[:])


tracks = np.array(tracks)
particles = np.array(particle_set)


np.save("data/tracks",tracks)
np.save("data/particles",particles)



# meanvector = tracker.track(scan)
