from scan import Scan, Particleset
import numpy as np 
from glob import glob
from filepath import Filepath
from os.path import join

lazfile_path = "/home/dunbar/Research/helheim/data/lazfiles"
lazfiles = glob(join(lazfile_path,"*.laz"))
lazfiles = [Filepath(file) for file in lazfiles]
lazfiles.sort(key=lambda i: i.datetime)

xbounds,ybounds = (535400.00+10,536400.00+10),(7358200.00+10,7359800+10)

for filepath in lazfiles:
	scan = Scan(filepath)
	orig = np.max(scan.file.points.shape)
	print(f"\n Down-sampling {scan.filepath.filepath}\n")
	points = np.squeeze(scan.bounds(xbounds,ybounds))
	shuffleinds = np.random.shuffle(np.arange(np.max(points.shape)))
	points = np.array(points)[shuffleinds]
	points = np.squeeze(points[::2].transpose())
	print(f"\n Reduction: {points.shape/orig}\n")
	scan.writefile(scan.filepath.filepath.replace(".laz",".dslaz"),points)

