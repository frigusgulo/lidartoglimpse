import numpy as np 
import matplotlib.pyplot as plt 

particleset = np.load("data/particles.npy",allow_pickle=True)[:,0,:,:]
print(particleset.shape)
start = (np.mean(particleset[0,:,0]),np.mean(particleset[0,:,1]))
cm = plt.cm.get_cmap('RdYlBu')
for i in range(particleset.shape[0]):
	x = particleset[i,:,0]
	y = particleset[i,:,1]
	x -= start[0]
	y -= start[1]
	vels = np.linalg.norm(particleset[i,:,3:5],axis=-1)
	#colors = np.unique(vels)
	#print(vels)
	#vels = np.searchsorted(colors,vels)
	plt.xlim(-3000,3000)
	plt.ylim(-3000,3000)
	plt.title("Particle Velocities M/D")
	sc = plt.scatter(x,y,s=5,c=vels,vmin=0,vmax=2,cmap=cm)
	plt.scatter(0,0,c='g',s=50)
	plt.colorbar(sc)
	plt.show()