# Attributes and methods for each algorithm

## Attributes
parameters: probably a dictionary, that will hold string as keys and (mostly) numpy matrices as values

## Methods
init: takes as input a dictionary specifying the hyperparameters of the algorithm (key values will have to be clarified in the documentation)
step: takes as input a data point (most likely a vector or tuple of vectors) and updates the parameters
transform: takes as input a data point and returns its (learned) representation
loss: takes as input a matrix (several data points) and returns the aggregate loss (a float) over the data points
