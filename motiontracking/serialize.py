import numpy as np
import laspy as lp 
import os
from os.path import join
import datetime
from glob import glob
from motiontracking.filepath import Filepath
from motiontracking.scanset import Scanset

lazfile_path = "/home/dunbar/Research/helheim/data/lazfiles"
lazfiles = glob(join(lazfile_path,"*.laz"))
lazfiles = [ Filepath(x) for x in lazfiles ]
scanset = Scanset(lazfiles)
scanset.serialize()

