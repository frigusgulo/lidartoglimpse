from . import *
from scan2dem import Scantodem
from filepath import Filepath
from glob import glob
from os.path import join
import numpy as np 
from shapely.geometry import box 
import geopandas as gpd
import json
import rasterio
from rasterio import mask

def bound_box(pointa,pointb):
	minx,miny = np.minimum(pointa[0],pointb[0]), np.minimum(pointa[1],pointb[1])
	maxx,maxy = np.maximum(pointa[0],pointb[0]), np.maximum(pointa[1],pointb[1])
	bbox = box(minx,miny,maxx,maxy)
	geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0],crs=32624) #24N
	return [json.loads(geo.to_json())['features'][0]['geometry']]

DEM_path = "/home/dunbar/Research/helheim/data/interpolated_dem"
lazfile_path = "/home/dunbar/Research/helheim/data/lazfiles"
raster_path = "/home/dunbar/Research/helheim/data/2016_cpd_vels"
lazfiles = glob(join(lazfile_path,"*.laz"))
velorasters = glob(join(raster_path,"*.tif"))
lazfiles = [ Filepath(x) for x in lazfiles ]
lazfiles.sort(key= lambda i: i.datetime)
cpdvels = [Filepath(x) for x in velorasters]
cpdvels.sort(key = lambda i: i.datetime)
diagupp, diaglow = (535000.00,7359250.00),(537000.00,7358250.00)
boundpolygon = bound_box(diagupp,diaglow)



with rasterio.open(cpdvels[0].filepath) as src:
	outimage = np.squeeze(mask.mask(src,boundpolygon,crop=True,filled=False)[0]) #extract mask
	
eastdims = (np.minimum(diagupp[0],diaglow[0]),np.maximum(diagupp[0],diaglow[0]),outimage.shape[1])
northdims = (np.minimum(diagupp[1],diaglow[1]),np.maximum(diagupp[1],diaglow[1]),outimage.shape[0])
easting_interpolation = np.linspace(eastdims[0],eastdims[1],int(eastdims[2]))
northing_interpolation = np.linspace(northdims[0],northdims[1],int(northdims[2]))

DEM_interp = Scantodem(lazfiles[0],DEM_path,32624)
DEM_interp.interp(bounds=[easting_interpolation,northing_interpolation])

import matplotlib.pyplot as plt 
fig,ax = plt.subplots(1,2)
ax[0].imshow(DEM_interp.bands[0])
ax[1].imshow(DEM_interp.bands[1])
#fig.colorbar()

plt.show()