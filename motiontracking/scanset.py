# Observer class

import datetime
import numpy as np 
from .scan import Scan 
from concurrent.futures import ThreadPoolExecutor
import dill
import os
from os.path import splitext
import sklearn
import sys
import glimpse
from motiontracking.filepath import Filepath
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
		images: list = None,
		skipinterval: int=2,
		load: bool = False):

		self.scans = scans
		self.skipinterval = skipinterval
		self.load = load
		self.images = images

		if len(scans) < 2:
			raise ValueError("Scans are not two or greater")
		if any(scan.datetime is None for scan in self.scans):
			raise ValueError(f"Scan {i} is missing datetime")
		#self.scans_dt = [scan.datetime for scan in self.scans].sort(lambda key x: x.datetime)
		if images is not None:
			self.scans.extend(self.images)
		self.observations = self.scans.copy()
		self.observations.sort(key = lambda x: x.datetime)
		self.date_times = [obs.datetime for obs in self.observations]
		time_deltas = np.array([dt.total_seconds() for dt in np.diff(self.date_times)])
		if any(time_deltas <= 0):
			raise ValueError("Image datetimes are not stricly increasing")
		self.datetimes = np.array(self.date_times)
		span = self.datetimes[0] - self.datetimes[-1]
		days = span.days
		print(f"\n Scanset Spans {days} Days\n")

	def get_inds(self,start_date,end_date):
		start = np.searchsorted(self.date_times,start_date)
		end = np.searchsorted(self.date_times,end_date)
		return np.arange(start,end+1)

	def obs_index(self,index):
		obs = self.observations[index]
		if isinstance(obs,Filepath):
			return self.scan_index(filepath=obs)
		elif isinstance(obs,glimpse.Image):
			#print("Using Image")
			return obs


	def scan_index(self,index=None,filepath=None,from_serial=True,build_tree=True) -> Scan:
		if index is not None:
			filepath = self.scans[index]
		elif filepath is not None:
			filepath = filepath
		pklfilepath = splitext(filepath.filepath)[0] + "_tree.pkl"
		if os.path.isfile(pklfilepath) and from_serial:
			try:
				print(f"\nInitializing Scan From {pklfilepath}\n")
				scan = Scan()
				with open(pklfilepath,'rb') as file:
					data = dill.load(file)
				
				scan.tree = data[0]
				scan.points = data[1]
				scan.datetime = data[2]

				return scan
			except:
				print(f"\nInitializing Scan From {filepath}\n")
				return Scan(filepath,build_tree)				

		else:
			print(f"\nInitializing Scan From {filepath}\n")
			return Scan(filepath,build_tree)

	def serialize(self):
		for scan in self.scans:
			filepath = splitext(scan.filepath)[0] + "_tree.pkl"
			if not os.path.isfile(filepath):
				try:
					scanobj = Scan(scan)
					toserial = [scanobj.tree,scanobj.points,scanobj.datetime]
					scanobj = None
					print(f"\nSerializing {filepath}\n")
					with open(filepath,'wb') as file:
						dill.dump(toserial,file)
				except:
					print(f"\n Unable to serialize {filepath}")
					#os.remove(filepath)
			