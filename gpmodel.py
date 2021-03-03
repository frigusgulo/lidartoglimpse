import torch 
import gpytorch
import torch.utils.data as Data
import pickle
from matplotlib import pyplot as plt
import numpy as np
from motiontracking.scan import Scan
from glob import glob
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


scan = Scan()
with open ("/home/dunbar/Research/helheim/data/lazfiles/160802_000217_tree.pkl","rb") as file:
	data = pickle.load(file)

scan.tree = data[0]
scan.points = data[1]
scan.datetime = data[2]
diagupp, diaglow = (535500.00,7358300.00),(536500.00,7359300)

eastdims = (np.minimum(diagupp[0],diaglow[0]),np.maximum(diagupp[0],diaglow[0])) 
northdims = (np.minimum(diagupp[1],diaglow[1]),np.maximum(diagupp[1],diaglow[1]))
easting = np.linspace(eastdims[0],eastdims[1],10).tolist()
northing = np.linspace(northdims[0],northdims[1],10).tolist()
grid = [[x,y] for x in easting for y in northing]

clusters = [scan.points[scan.query(point,radius=15)]   for point in grid]
#clusters = [(cluster - np.mean(cluster,axis=0))/np.std(cluster,axis=0) for cluster in clusters]
train_x = torch.tensor(clusters[1][:,:2],dtype=torch.float)
train_y = torch.tensor(clusters[1][:,-1],dtype=torch.float)

#train_x = torch.tensor(np.vstack([cluster[:,0:2] for cluster in clusters]))
#train_y = torch.tensor(np.vstack([cluster[:,-1:] for cluster in clusters]))


print("Length of training data: {}".format(len(train_x)))

# initialize likelihood and model
dataset = Data.TensorDataset(train_x,train_y)
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

#print(model.covar_module.ScaleKernel.outputscale())
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=1e-2)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
'''
loader = Data.DataLoader(
	dataset = dataset,
	batch_size=2**16,
	num_workers=12)
'''


training_iter = 1000

settings = gpytorch.settings.debug(state=False)

with settings:
	for i in range(training_iter):
		# Zero gradients from previous iteration
		#for step,(x,y) in enumerate(loader):
			 
		output = model(train_x)
		loss = -mll(output, train_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if i% 10 ==0:
			print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f   outputscale: %.3f' % (
			    i + 1, training_iter, loss.mean().item(),
			    model.covar_module.base_kernel.lengthscale.item(),
			    #model.covar_module.scale_kernel.outputscale.item(),
			    model.likelihood.noise.item(),
			    model.covar_module.raw_outputscale.item()
			))

for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.item()}')

	# Get into evaluation (predictive posterior) mode


# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
