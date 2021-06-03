from sklearn.neighbors import KDTree 
import numpy as np
import time

arr = np.random.random((100000,3))*10

for i in reversed(range(1,50)):
	start = time.time()
	tree = KDTree(arr,leaf_size=i)
	buildtime = time.time() - start

	start = time.time()
	tree.query(arr[0:100],k=150)[0]
	print(f"\n Build Time {buildtime}, Query Time {time.time()-start}\n")