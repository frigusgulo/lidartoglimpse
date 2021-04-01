import pdb
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import glimpse
import numpy as np
import datetime
import os
from shapely.geometry import LinearRing
import glob
import itertools
from os.path import join
import rasterio
os.environ['OMP_NUM_THREADS'] = '12'
from motiontracking.scanset import Scanset
from motiontracking.raster import Raster
from motiontracking.filepath import Filepath
from motiontracking.scan import Scan
from motiontracking.scanset import Scanset
#==============================================

lazfile_path = "/home/dunbar/Research/helheim/data/lazfiles"
lazfiles = glob.glob(join(lazfile_path,"*.laz"))
lazfiles = [ Filepath(x) for x in lazfiles ]
scanset = Scanset(lazfiles)

#scan = scanset.index(0,from_serial=False,build_tree=False)
#scan_hull = scan.convexhull().vertices

#scan_hull = scan.points[scan_hull,:2]
#scan_hull = list(scan_hull.simplices[:,:2].astype(np.int))


DATA_DIR = "/home/dunbar/Research/helheim/data/observations"
DEM_DIR = os.path.join(DATA_DIR, 'dem')
MAX_DEPTH = 30e3

# ---- Prepare Observers ----

observerpath = ['stardot1','stardot2']
observers = []
datetimestotal = []
for observer in observerpath:
    path = join(DATA_DIR,observer)
    campaths =  glob.glob(join(path,"*.JSON"))
    images = [glimpse.Image(path=campath.replace(".JSON",".jpg"),cam=glimpse.Camera.from_json(campath)) for campath in campaths]
    images.sort(key= lambda img: img.datetime)
    datetimes = np.array([img.datetime for img in images])
    datetimestotal.append(datetimes)
    '''
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

# Prepare DEM 
path = glob.glob(join(DEM_DIR,"*.tif"))[0]

print("DEM PATH: {}".format(path))
dem = glimpse.Raster.open(path=path)

dem.crop(zlim=(0, np.inf))
dem.fill_crevasses(mask=~np.isnan(dem.array), fill=True)
'''
scan_datetimes = matplotlib.dates.date2num(np.array(scanset.date_times))

cam_1_datetimes = matplotlib.dates.date2num(np.array(datetimestotal[0]))
cam_2_datetimes = matplotlib.dates.date2num(np.array(datetimestotal[1]))

max_date = np.max(scan_datetimes)
min_date = np.min(scan_datetimes)




cam_1_datetimes = cam_1_datetimes[cam_1_datetimes >= min_date]
cam_1_datetimes = cam_1_datetimes[cam_1_datetimes <= max_date]

cam_2_datetimes = cam_2_datetimes[cam_2_datetimes >= min_date]
cam_2_datetimes = cam_2_datetimes[cam_2_datetimes <= max_date]




span = max_date-min_date

#cam_1_datetimes = [day.hours for day in cam_1_datetimes.tolist()]
#domain = np.linspace(0,span.days,int(span.total_seconds()//(3600*12)))
matplotlib.pyplot.plot_date(cam_1_datetimes,np.ones_like(cam_1_datetimes),label="Camera 1 Samples")
matplotlib.pyplot.plot_date(cam_2_datetimes,2*np.ones_like(cam_2_datetimes),label="Camera 2 Samples")
matplotlib.pyplot.plot_date(scan_datetimes,3*np.ones_like(scan_datetimes),label="ATLAS Samples")
plt.yticks([1,2,3],labels=["Cam 1","Cam 2","ATLAS"])
plt.title("Sample Times For ATLAS and Both Cameras")
plt.grid(True)
plt.xlabel("Date")
plt.legend()
plt.show()

'''

# ---- Prepare viewshed ----
for obs in observers:
    dem.fill_circle(obs.images[0].cam.xyz, radius=50)
viewshed = dem.copy()
viewshed.array = np.ones(dem.shape, dtype=bool)
for obs in observers:
    viewshed.array &= dem.viewshed(obs.images[0].cam.xyz,correction=True)
points = dem.rasterize_poygons(scan_hull)
viewshed &= points
'''
'''
from shapely.geometry import mapping, Polygon
import fiona

viewshed_0 = Polygon(observers[0].images[0].cam.viewpoly(depth=8e3)[:,:2])
viewshed_1 = Polygon(observers[1].images[0].cam.viewpoly(depth=8e3)[:,:2])
scan_hull = Polygon(scan_hull[:,:2])

schema = {
    'geometry': 'Polygon',
    'properties': {'id': 'int'},
}

with fiona.open('cam_0.shp', 'w', 'ESRI Shapefile', schema) as c:
    ## If there are multiple geometries, put the "for" loop here
    c.write({
        'geometry': mapping(viewshed_0),
        'properties': {'id': 123},
    })
with fiona.open('cam_1.shp', 'w', 'ESRI Shapefile', schema) as c:
    ## If there are multiple geometries, put the "for" loop here
    c.write({
        'geometry': mapping(viewshed_1),
        'properties': {'id': 123},
    })

with fiona.open('scan_bounds.shp', 'w', 'ESRI Shapefile', schema) as c:
    ## If there are multiple geometries, put the "for" loop here
    c.write({
        'geometry': mapping(scan_hull),
        'properties': {'id': 123},
    })
'''
'''
import rasterio
src = rasterio.open(path)
raster = src.read(1)
raster[raster==np.min(raster)] = None



row,col = src.index(scan_hull[:,0],scan_hull[:,1])
locs = np.vstack([row,col]).T[:,::-1]
horizon_0 = np.vstack([point for point in dem.horizon(observers[0].images[0].cam.xyz)]).tolist()
horizon_1 = np.vstack([point for point in dem.horizon(observers[1].images[0].cam.xyz)]).tolist()
horizon_0 = np.vstack([src.index(point[0],point[1]) for point in horizon_0])
horizon_1 = np.vstack([src.index(point[0],point[1]) for point in horizon_1])



plt.imshow(raster)
plt.plot(locs[:,0],locs[:,1],'k-')

plt.plot(horizon_0[:,1],horizon_0[:,0],'r-')
plt.plot(horizon_1[:,1],horizon_1[:,0],'b-')
plt.show()
plt.savefig("/home/dunbar/Research/helheim/viewshedplot")

'''