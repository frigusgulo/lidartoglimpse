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
import matplotlib.pyplot as plt


from motiontracking.motion import CartesianMotion
from motiontracking.scanset import Scanset
from motiontracking.raster import Raster
from motiontracking.filepath import Filepath
from motiontracking.scan import Scan, Particleset
from motiontracking.scanset import Scanset
from motiontracking.tracker import Tracker


raster_path = "./data/2016_cpd_vels"
#lazfile_path = "./data/lazfiles"
lazfile_path = "./data/downsampledlazfiles"
velorasters = glob(join(raster_path,"*.tif"))
lazfiles = glob(join(lazfile_path,"*.dslaz"))
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
grid = [grid[mid]]
#grid[0][1]+=1000
#grid[0][0]+=500

initialscan = Scan(lazfiles[0])
scanset = Scanset(lazfiles[1:])

# CPD resolution is 15x15 meters per cell

DEM = Raster("./data/DEM/helheim_wgs84UTM.tif")
motionparams = {
	"timestep": datetime.timedelta(days=1),
	"DEM": DEM,
	"xy_sigma": np.array([0.1,0.1]),
	"vxyz": np.array([0,0,0]),
	"vxyz_sigma": np.array([10.0,10.0,0.1]),
	"axyz": np.array([0,0,0]),
	"axyz_sigma": np.array([3.0,3.0,0.02]),
        'n':250
}


mm = CartesianMotion(xy=grid[0],**motionparams)

trackergrid = [Tracker(CartesianMotion(xy=point,**motionparams),initialscan) for point in grid]
print(f"\n Observing At {len(trackergrid)} Points \n")
tracker=trackergrid[0]

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances as dist

s_init = scanset.index(0)

query_radius = 15
x_ref = tracker.particles.mean(axis=0)[:2]
skip=1

ref_points = s_init.points[s_init.tree.query_ball_point(x_ref,query_radius)][::skip]
train_size = min(1000,ref_points.shape[0])
rand_ind = np.random.choice(range(len(ref_points)),train_size,replace=False)
ref_points = ref_points[rand_ind]

l = 0.1
sigma2 = 0.01
r_mean = ref_points.mean(axis=0)
r_std = ref_points.std(axis=0)
x_train = (ref_points - r_mean)/r_std
K = np.exp(-dist(x_train[:,:2],x_train[:,:2],squared=True)/l**2) + sigma2*np.eye(x_train.shape[0])
Kinv = np.linalg.inv(K)
particles_init = tracker.particles.copy()

init_index = np.linspace(0,len(particles_init)-1,len(particles_init)).astype(int)

particle_list = []
mean_list = []

fig,axs = plt.subplots()
fig.set_size_inches(12,12)
s0 = scanset.index(0)
particles_0 = tracker.particles.copy()
w = np.ones(len(particles_0))
w/=w.sum()

plt.scatter(particles_0[:,0],particles_0[:,1],c=w)

particle_list.append(particles_0.copy())
mean_list.append(particles_0.mean(axis=0))

for i in range(10):
    print(i)
    s1 = scanset.index(i+1)
    dt = s1.datetime - s0.datetime

    particles_next = tracker.motionmodel.evolve_particles(particles_0.copy(),dt)

    delta_0 = particles_init[init_index,:2] - x_ref
    delta_p = particles_next[:,:3] - particles_init[init_index,:3] 

    query_indices = s1.tree.query_ball_point(particles_next[:,:2] - delta_0,query_radius)

    query_clouds = [(s1.points[i] - delta_p[k]) 
        for k,i in enumerate(query_indices)]

    query_lls = np.zeros((len(query_clouds)))
    test_size = min(min(len(q) for q in query_clouds),100)

    for j,query_points in enumerate(query_clouds):
        print(j)
        rand_ind = np.random.choice(range(len(query_points)),test_size,replace=False)
        x_test = (query_points[rand_ind] - r_mean)/r_std
        Kstar = np.exp(-dist(x_test[:,:2],x_train[:,:2],squared=True)/l**2)
        Kss = np.exp(-dist(x_test[:,:2],x_test[:,:2],squared=True)/l**2)

        mu = Kstar @ Kinv @ x_train[:,2]
        Sigma = Kss - Kstar @ Kinv @ Kstar.T + sigma2*np.eye(Kss.shape[0])

        log_like = -0.5*(np.log(2*np.pi)*x_test.shape[0] + np.linalg.slogdet(Sigma)[1] + 0.5*(x_test[:,2]-mu)@np.linalg.inv(Sigma)@(x_test[:,2]-mu))
        query_lls[j] = log_like

    w = np.exp(0.2*(query_lls-query_lls.max()))
    w/=w.sum()
    ind = np.random.choice(range(len(particles_next)),len(particles_next),p=w)
    color = np.array([plt.cm.jet(i/10)]*len(particles_next))
    color[:,-1] = w/w.max()
    plt.scatter(particles_next[:,0],particles_next[:,1],color=color)

    particles_0 = particles_next[ind]
    init_index = init_index[ind]
    s0 = s1

    particle_list.append(particles_0.copy())
    mean_list.append(particles_0.mean(axis=0))
    





'''

Un comment for calibration

calibration_iters = 24
timestep = np.ones(calibration_iters)
timestep = timestep.tolist()

particle_set = []
tracker = trackergrid[0]
for time in timestep:
	dt = datetime.timedelta(hours=time)
	particles = tracker.track(calibrate=True,dt=dt)
	particle_set.append(particles)

print(len(particle_set))
tracks = np.array(tracks)
particle_set = np.array([ps for ps in particle_set])

print(tracks.shape)
print(np.unique(particles,axis=0).shape)


np.save("data/calibration_particles",particle_set)
print(f"\n\n Calibration Complete\n\n")
'''
"""
tracks = []
particle_set = []
for i in range(1,5):


	testscan = scanset.index(i)
	means,particles = zip(*[tracker.track(scan=testscan) for tracker in trackergrid])
	tracks.append(means[:])
	particle_set.append(particles[:][0].squeeze())


tracks = np.array(tracks)
particles = np.array(particle_set)


np.save("data/tracks",tracks)
np.save("data/particles",particles)



# meanvector = tracker.track(scan)
"""
