# cynfrence
A cython based inference backend for keras ( ... in the making ). 

# motivation
Keras is great to train. Keras backend ( other than theano which is no longer maintained ) are hard to install on target computers ( especially arm computers ). They are also not optimized out of the box. 

cynfrence is meant to be a super light weight set of code that can automatically turn keras models into cython compiled .so ( or pyc for windows ) files that run the network with almost no overhead. 

Since cynfrence will be all cython it will easily compile on all OS's with optimizations appropriate for the machine. 

# How it works. 
The vision : load a keras model and it will be automatically parsed and reconstructed as a pyx + setup.py file and compiled into an so. 
The weights will be hard coded, the layers will be super specific and so have very little overhead. This will allow for possibly the fastest inference system available on cpu only computers ( especially arm computers ( IoT )). 

Current state: 
1) There are 4 layers available currently that allow the inference of fully 2d convolutional networks 
* conv2d_no_padding
* conv2d_with_padding
* seperable_conv2d_no_padding
* seperable_conv2d_with_padding
Strides and dilations and supported ( somewhat tested ).
All standards follow tensorflow for compatibility. 

# TODO :

Here is where i would very much appreciate your help... ;)
short term:
1) create thorough tests. 
2) add basic layers like batch normalization, pooling ( average, max ), dense, activations. 
* numpy / scipy inplementations likely already exist and are super fast as is... 
3) Create a simple translator from keras models to cython using the available layers. 

longer term:
1) add all keras layers with full features support. 
* 1D, 3D, transpose Convolutions.
* Reccurent Layers ( GRU, LSTM, simpleRNN... )
2) make the installation super simple and pain free while keeping very high performance.

long long term:
1) add specialized optimizations such as fft based convolutions when kernel is > 7x7 ( or even run a set of tests on the machine in order to evaluate which optimizations are best to use ). 
2) fork keras and add cynfrence as a backend with all the learning capabilities turned off. 
