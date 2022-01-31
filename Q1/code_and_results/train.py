'''
	Reference:
	The code is adapted from https://github.com/onermustafaumit/SRTPMs/blob/main/LUAD/mil_dpf_regression/train.py
'''

import os
import sys
import numpy as np
from model import Model
import torch
import torch.utils.data
from load_data import load
from tqdm import tqdm
import matplotlib.pyplot as plt

parameters = {'patch_size':28,
			  'batch_size':100,
			  'num_classes':1,
			  'num_instances':100,
			  'num_features':28,
			  'num_bins':21,
			  'sigma':0.05,
			  'learning_rate':1e-4,
			  'weight_decay':0.0005,
			  'num_epochs':30,
			  'metrics_file':'loss/loss_metrics.txt'}

# load the training and testing data as dataloaders
print("Load the training and testing data")
train_data_loader, test_data_loader = load()

# construct model
# use GPU if avaliable, otherwise, cpu
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=parameters["num_classes"],
			  num_instances=parameters["num_instances"],
			  num_features=parameters["num_features"],
			  num_bins=parameters["num_bins"],
			  sig=parameters["sigma"])
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])

# print model parameters
print('# Model parameters:')
for k in parameters:
	print(k + ": " + str(parameters[k]))

# write model parameters into metrics file
with open(parameters['metrics_file'],'w') as f_metrics_file:
	f_metrics_file.write('# Model parameters:\n')
	for key in parameters:
		f_metrics_file.write('# {} = {}\n'.format(key, parameters[key]))
	f_metrics_file.write('# epoch\ttraining_loss\tvalidation_loss\n')

# Use the same loss function as the author
criterion = torch.nn.L1Loss()

# start iterating
train_loss_ls = []
val_loss_ls = []
for epoch in range(parameters['num_epochs']):
	training_loss = 0
	validation_loss = 0
	# train for one epoch
	print('Start the training for epoch: {}'.format(epoch+1))
	num_predictions = 0
	model.train()
	progress_bar = tqdm(range(len(train_data_loader)))
	for images, targets in train_data_loader:
		images = images.to(device)
		targets = targets.to(device)
		# zero the parameter gradients
		optimizer.zero_grad()
		# forward + backward + optimize
		output = model(images)
		loss = criterion(output, targets)
		loss.backward()
		optimizer.step()
		training_loss += loss.item()*targets.size(0)
		num_predictions += targets.size(0)
		progress_bar.update(1)
	training_loss /= num_predictions
	progress_bar.close()
	# evaluate on the validation dataset
	print('Validation')
	num_predictions = 0
	model.eval()
	progress_bar = tqdm(range(len(test_data_loader)))
	with torch.no_grad():
		for images, targets in test_data_loader:
			images = images.to(device)
			targets = targets.to(device)
			# forward
			output = model(images)
			loss = criterion(output, targets)
			validation_loss += loss.item()*targets.size(0)
			num_predictions += targets.size(0)
			progress_bar.update(1)
	validation_loss /= num_predictions
	progress_bar.close()
	print('Epoch='+str(epoch+1)+': training_loss='+str(round(training_loss,2))+', validation_loss='+str(round(validation_loss,2)))
	# add to list for ploting
	train_loss_ls.append(training_loss)
	val_loss_ls.append(validation_loss)
	# logging loss values into metrics file
	with open(parameters['metrics_file'],'a') as f_metrics_file:
		f_metrics_file.write(str(epoch+1)+'\t'+str(round(training_loss,2))+'\t'+str(round(validation_loss,2))+'\n')


model_weights_filename = 'model/model_weights_{}.pth'.format(epoch+1)
state_dict = {	'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}
torch.save(state_dict, model_weights_filename)
print("Model weights saved in file: ", model_weights_filename)
print('Training finished!!!')


plt.plot(train_loss_ls, label = 'train loss')
plt.plot(val_loss_ls, label = 'validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss for Training and Validation')
plt.legend()
plt.savefig("loss.png")
