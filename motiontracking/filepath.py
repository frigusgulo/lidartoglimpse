import numpy as np 
import datetime
from os.path import splitext, basename
class Filepath:
	def __init__(self,
		filepath):


		self.filepath = filepath
		filename =  splitext( basename(filepath) )[0].replace("_","")
		dateobj = [int(filename[i:i+2]) for i in range(0,len(filename),2)] # year, month,day,hour,min,sec

		self.datetime = datetime.datetime(dateobj[0],dateobj[1],dateobj[2],dateobj[3],dateobj[4],dateobj[5],0)	
