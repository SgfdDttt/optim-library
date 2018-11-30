# inside this directory, run 'make -f make_mnist_data.mk'
script=scripts/mnist2txt.py

all: mnist.txt

train-images-idx3-ubyte:
	wget yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O $@.gz
	gunzip $@.gz

mnist.txt: train-images-idx3-ubyte
	python $(script) $< $@
