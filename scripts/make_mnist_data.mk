# inside this directory, run 'make -f make_mnist_data.mk'
script=mnist2csv.py

all: train_mnist.csv test_mnist.csv

train-images-idx3-ubyte:
	wget yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O $@.gz
	gunzip $@.gz

t10k-images-idx3-ubyte.gz:
	wget yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O $@.gz
	gunzip $@.gz

train_mnist.csv: train-images-idx3-ubyte
	python $(script) $< $@
	
test_mnist.csv: t10k-images-idx3-ubyte.gz
	python $(script) $< $@
