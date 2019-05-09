#! /bin/sh

set -e

mkdir -p pretrained

wget http://homes.cs.washington.edu/~haichen/models/O1.prototxt -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/O1.caffemodel -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/O2.prototxt -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/O2.caffemodel -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/alexnet_mean.binaryproto -NP pretrained/

wget http://homes.cs.washington.edu/~haichen/models/S1.prototxt -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/S1.caffemodel -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/S2.prototxt -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/S2.caffemodel -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/places205_mean.binaryproto -NP pretrained/

wget http://homes.cs.washington.edu/~haichen/models/vgg_face_msr201.prototxt -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/vgg_face_msr201.caffemodel -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/F1.prototxt -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/F1.caffemodel -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/F2.prototxt -NP pretrained/
wget http://homes.cs.washington.edu/~haichen/models/F2.caffemodel -NP pretrained/
