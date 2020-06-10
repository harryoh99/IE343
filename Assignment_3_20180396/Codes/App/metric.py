import numpy as np
# Calculate the Gini index for a split dataset
def gini_index(groups):
	'''
		Gini index

		Args:
			children groups : (left, right) dataset [n_datapoints, n_features + 1], [n_datapoints, n_features + 1]

		Returns:
			gini : gini index (scalar)
	'''
	
	# count all samples at split point
	n_instances = np.sum([np.shape(group)[0] for group in groups])
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(np.shape(group)[0])
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0	
		# score the group based on the score for each class (ref : np.unique)
		classes = list(np.unique(group[:,-1]))
		for i in classes:
			proportion = np.sum(group[:,-1]==i)/size
			score += proportion*proportion
		# Exceptional control 
		gini += (1.0 - score) * (size / n_instances)
		assert gini >= 0		
	return gini