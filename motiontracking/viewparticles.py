import numpy as np 
import matplotlib.pyplot as plt 

particleset = np.load("data/calibration_particles.npy",allow_pickle=True)
print(particleset.shape)
print(np.unique(particleset,axis=0).shape)
start = (np.mean(particleset[0,:,0]),np.mean(particleset[0,:,1]))
cm = plt.cm.get_cmap('RdYlBu')
for i in range(particleset.shape[0]):
	x = particleset[i,:,0]
	y = particleset[i,:,1]

	print(f"\n Variance X {np.var(x)}, Variance Y {np.var(y)}\n")
	x -= start[0]
	y -= start[1]
	vels = np.linalg.norm(particleset[i,:,3:5],axis=-1)
	#colors = np.unique(vels)
	#print(vels)
	#vels = np.searchsorted(colors,vels)
	plt.xlim(-300,300)
	plt.ylim(-300,300)
	plt.title("Particle Velocities M/D")
	sc = plt.scatter(x,y,s=5,c=vels,vmin=-20,vmax=20,cmap=cm)
	plt.scatter(0,0,c='g',s=50)
	plt.colorbar(sc)
	plt.show()
	plt.clf()