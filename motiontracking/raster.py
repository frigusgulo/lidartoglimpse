import rasterio
import numpy as np
import matplotlib.pyplot as plt
class Raster:

	def __init__(self,
		demfilepath: str = None
		):
	
		self.dem = rasterio.open(demfilepath,'r')

	def getval(self,xy_locs: np.ndarray,band: int=1):
		# TODO NEED TO VECTORIZE! 
	
		vals = []
		for i in range(xy_locs.shape[0]):
			#print(index[i,:])
			
			index = self.index(xy_locs[i,:])
			value = self.dem.read(band)[index]
			vals.append(value)
	

		#print(index[np.argmax(row),0],index[np.argmax(col),1]) # DEBUG
		return vals

	def index(self,index: np.ndarray):
		return self.dem.index(index[0],index[1])