"""
    Reference:
    To make sure the architecture of the model is the same as the author used as required in the question,
    the following code for defining the model and pooling layer is adpated from the source code:
    https://github.com/onermustafaumit/SRTPMs/blob/main/LUAD/mil_dpf_regression/model.py
    and
    https://github.com/onermustafaumit/SRTPMs/blob/main/LUAD/mil_dpf_regression/distribution_pooling_filter.py
"""
import math
import torch
import torch.nn as nn
from resnet import resnet18

####### Define models

# Define the feature extractor made of resnet18
class Feature_Extractor(nn.Module):
    def __init__(self, num_features=28):
        super(Feature_Extractor, self).__init__()
        # use resnet18
        self.model = resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_features)
        # ReLu
        self.relu = nn.ReLU()

    def forward(self, x):
        res_out = self.model(x)
        output = self.relu(res_out)
        return output

# The pooling layer
class Pooling(nn.Module):
	__constants__ = ['num_bins', 'sig']
	def __init__(self, num_bins=21, sig=0.0167):
		super(Pooling, self).__init__()
		self.num_bins = num_bins
		self.sig = sig
		self.alfa = 1/math.sqrt(2*math.pi*(sig**2))
		self.beta = -1/(2*(sig**2))
		sample_points = torch.linspace(0,1,steps=num_bins, dtype=torch.float32, requires_grad=False)
		self.register_buffer('sample_points', sample_points)

	def extra_repr(self):
		return 'num_bins={}, sig={}'.format(
			self.num_bins, self.sig
		)

	def forward(self, data):
		batch_size, num_instances, num_features = data.size()
		sample_points = self.sample_points.repeat(batch_size,num_instances,num_features,1)
		# sample_points.size() --> (batch_size,num_instances,num_features,num_bins)
		data = torch.reshape(data,(batch_size,num_instances,num_features,1))
		# data.size() --> (batch_size,num_instances,num_features,1)
		diff = sample_points - data.repeat(1,1,1,self.num_bins)
		diff_2 = diff**2
		# diff_2.size() --> (batch_size,num_instances,num_features,num_bins)
		result = self.alfa * torch.exp(self.beta*diff_2)
		# result.size() --> (batch_size,num_instances,num_features,num_bins)
		out_unnormalized = torch.sum(result,dim=1)
		# out_unnormalized.size() --> (batch_size,num_features,num_bins)
		norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
		# norm_coeff.size() --> (batch_size,num_features,1)
		output = out_unnormalized / norm_coeff
		# out.size() --> (batch_size,num_features,num_bins)
		return output

# Define the representation transformation MLP
class Re_Trans(nn.Module):
    def __init__(self, num_features=32, num_bins=11, num_classes=10):
        super(Re_Trans, self).__init__()
        # three layered MLP
        self.mlp = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features * num_bins, 384),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(384, 192),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(192, num_classes)
        )

    def forward(self, x):
        output = self.mlp(x)
        return output


# The whole model
class Model(nn.Module):
    def __init__(self, num_classes=10, num_instances=100, num_features=32, num_bins=11, sig=0.1):
        super(Model, self).__init__()
        self._num_classes = num_classes
        self._num_instances = num_instances
        self._num_features = num_features
        self._num_bins = num_bins
        self._sig = sig
        # feature extractor module
        self._feature_extractor = Feature_Extractor(num_features=num_features)
        # Pooling
        self._pooling = Pooling(num_bins=num_bins, sig=sig)
        # bag-level representation transformation module
        self._representation_transformation = Re_Trans(num_features=num_features, num_bins=num_bins, num_classes=num_classes)

    def forward(self, x):
        fe_out = self._feature_extractor(x)
        reshape_out = torch.reshape(fe_out,(-1,self._num_instances,self._num_features))
        pool_out = self._pooling(reshape_out)
        flat_out = torch.flatten(pool_out, 1)
        output = self._representation_transformation(flat_out)
        return output

# model = Model()
# print(model)
