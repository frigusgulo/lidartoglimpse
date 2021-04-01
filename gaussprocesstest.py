import numpy as np 
from matplotlib import pyplot as plt
import csv
from sklearn.metrics.pairwise import euclidean_distances as dist


def normalize(data,mean,std):
	return (data - mean)/std


def kernel(x,y,lengthscale):
	return np.exp(-dist(x,y,squared=True)/(lengthscale**2)) + 1e-300


useable = []
with open('../daily_air_turnagain.txt',newline='') as file:
	data = csv.reader(file,delimiter=',')
	add = False
	for row in data:
		if (len(row) == 2) and row[0] == 'Date':
			add = True
		elif add:
			useable.append(row)

y_train = np.array(useable)[:,1]
dates = np.array(useable)[:,0]

dataspan = y_train.shape[0] - 365
y_train = y_train[dataspan:]
dates = dates[dataspan:]

mask = np.argwhere(y_train=="")

x_train = np.linspace(0,y_train.shape[0]*24,y_train.shape[0])
x_interp = np.linspace(0,x_train.shape[0]*24,x_train.shape[0]*2)[:,np.newaxis] # Interpolate at 8 hour resolution from daily temps



x_train = np.delete(x_train,mask)
y_train = np.delete(y_train,mask)
y_train = y_train[:,np.newaxis]
x_train = x_train[:,np.newaxis]


x_train = x_train[::1,:].astype(np.float32)
y_train = y_train[::1,:].astype(np.float32)

refmeany = np.mean(y_train)
refstdy = np.std(y_train)
refmeanx = np.mean(x_train)
refstdx = np.std(x_train)





x_train_norm = normalize(x_train,refmeanx,refstdx)
y_train_norm = normalize(y_train, refmeany,refstdy)

x_interp_norm = normalize(x_interp,refmeanx,refstdx)


lengthscale =1e-2
obsvar = 5e-2



K_train = kernel(x_train_norm,x_train_norm,lengthscale=lengthscale).astype(np.float32)
K_star = kernel(x_interp_norm,x_train_norm,lengthscale=lengthscale).astype(np.float32)
K_star_star = kernel(x_interp_norm,x_interp_norm,lengthscale=lengthscale).astype(np.float32)

K_train = np.linalg.inv(K_train + np.eye(K_train.shape[0])*obsvar)

covariance = K_star_star - K_star@K_train@K_star.T + obsvar*np.eye(K_star_star.shape[0])
mu = K_star@K_train@y_train



stddev = np.sqrt(np.diagonal(K_star_star))
x_interp = np.squeeze(x_interp)
mu = np.squeeze(mu)



ticks = x_interp[::2]
labels = dates 

ticks = ticks[5::14]
labels = labels[5::14]

annotation = "Data From:  https://wcc.sc.egov.usda.gov/nwcc/rgrpt?report=daily_tavg_por&state=AK&operation=View"

plt.plot(x_interp,mu,'k',lw=1,label='Prediction Mean')
plt.fill_between(x_interp,mu-2*stddev,mu+2*stddev,alpha=0.4,color='g',label="99% Confidence Interval")
plt.scatter(x_train,y_train,c='r',s=6,label="Observation Points")
plt.xticks(ticks=ticks,labels=labels,rotation=30)
plt.xlabel("Date")
plt.ylabel("Air Temperature (Degrees Farenheit)")
plt.title("Daily Mean Air Temperature (deg F) at Turnagain Pass, AK Collected From That Fancy Snotel Site",pad=15,loc='left',fontsize=20)
plt.ylim(-10,75)
plt.legend()
plt.text(-2, -5, annotation, ha='left', fontsize = 7, alpha=0.9)
plt.show()
