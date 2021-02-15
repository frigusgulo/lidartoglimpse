from motiontracking.scan import Scan, Particleset
import numpy as np 
from glob import glob
from motiontracking.filepath import Filepath
from os.path import join,basename

lazfile_path = "/home/dunbar/Research/helheim/data/lazfiles"
lazfiles = glob(join(lazfile_path,"*.laz"))
lazfiles = [Filepath(file) for file in lazfiles]
lazfiles.sort(key=lambda i: i.datetime)

xbounds,ybounds = (535400.00-100,536400.00+100),(7358200.00-100,7359800+100)

for filepath in lazfiles:
	scan = Scan(filepath)
	
	print(f"\n Down-sampling {scan.filepath.filepath}\n")


	dest = join("/home/dunbar/Research/helheim/lidartoglimpse/data/downsampledlazfiles",basename(scan.filepath.filepath).replace(".laz",".dslaz"))
	scan.downsample(dest,5,xbounds,ybounds)
	


