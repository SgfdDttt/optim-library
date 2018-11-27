datadir=mnist
script=scripts/mnist2txt.py
$(shell mkdir -p $(datadir))

all: $(datadir)/mnist.txt

$(datadir)/train-images-idx3-ubyte:
	wget yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O $@.gz
	gunzip $@.gz

$(datadir)/mnist.txt: $(datadir)/train-images-idx3-ubyte
	python $(script) $< $@
