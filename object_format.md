# Attributes and methods for each algorithm

At the top of each class file, add the reference to a paper where that algorithm is described. The notation used for the parameters and the hyperparameters should be as close as possible to the notation used in that paper.

## Attributes
- parameters: probably a dictionary, that will hold string as keys and (mostly) numpy matrices as values
- hyperparameters (optional): other dictionary to clearly separate the hyperparameters (passed to the alogorithms upon initialization) from the parameters (initialized from the hyperparameters and/or updated during the algorithm steps)

## Methods
- init: takes as input a dictionary specifying the hyperparameters of the algorithm (key values will have to be clarified in the documentation)
- step: takes as input a data point (most likely a vector or tuple of vectors) and updates the parameters
- transform: takes as input n data points as an (n x d) matrix, and returns their (learned) representation as an (n x k) matrix
- loss: takes as input n data points as an (n x d) matrix and returns the aggregate loss (a float) over the data points
