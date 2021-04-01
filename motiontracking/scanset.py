# Observer class

import datetime
import numpy as np 
from .scan import Scan 
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
from os.path import splitext
from sklearn.neighbors import KDTree 
class Scanset(Scan):
	'''
	A sequence of LiDAR scan observations

	Attributes:
		scans (list) : list of filepaths to LiDAR point clouds stored in scrictly increasing time

		density (int) : skip Interval of points to store in the KD tree

		load (bool): Weather to load all scans into memory or untill queried 
	

	'''

	def __init__(
		self,
		scans: list,
		skipinterval: int=2,
		load: bool = False):

		self.scans = scans
		self.skipinterval = skipinterval
		self.load = load

		if len(scans) < 2:
			raise ValueError("Scans are not two or greater")
		if any(scan.datetime is None for scan in self.scans):
			raise ValueError(f"Scan {i} is missing datetime")
	
		self.scans.sort(key = lambda x: x.datetime)
		self.date_times = [scan.datetime for scan in self.scans]
		time_deltas = np.array([dt.total_seconds() for dt in np.diff(self.date_times)])
		if any(time_deltas <= 0):
			raise ValueError("Image datetimes are not stricly increasing")
		self.datetimes = np.array(self.date_times)
		span = self.datetimes[0] - self.datetimes[-1]
		days = span.days
		print(f"\n Scanset Spans {days} Days\n")


	def index(self,index,from_serial=True,build_tree=True) -> Scan:
		filepath = self.scans[index]
		pklfilepath = splitext(filepath.filepath)[0] + "_tree.pkl"
		if os.path.isfile(pklfilepath) and from_serial:
			print(f"\nInitializing Scan From {pklfilepath}\n")
			scan = Scan()
			with open(pklfilepath,'rb') as file:
				data = pickle.load(file)
			
			scan.tree = data[0]
			scan.points = data[1]
			scan.datetime = data[2]
			return scan
		else:
			print(f"\nInitializing Scan From {filepath}\n")
			return Scan(filepath,build_tree)

	def serialize(self):
		for scan in self.scans:
			filepath = splitext(scan.filepath)[0] + "_tree.pkl"
			if not os.path.isfile(filepath):
				scanobj = Scan(scan)
				toserial = [scanobj.tree,scanobj.points,scanobj.datetime]
				scanobj = None
				print(f"\nSerializing {filepath}\n")
				with open(filepath,'wb') as file:
					pickle.dump(toserial,file)
	
			