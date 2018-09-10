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
-[ ] MaxPooling2D
-[ ] AveragePooling2D

3) Create a simple translator from keras models to cython using the available layers. 

longer term:
1) add all keras layers with full features support. 
* 1D, 3D, transpose Convolutio
* Reccurent Layers ( GRU, LSTM, simpleRNN... )
2) make the installation super simple and pain free while keeping very high performance.

long long term:
1) add specialized optimizations such as fft based convolutions when kernel is > 7x7 ( or even run a set of tests on the machine in order to evaluate which optimizations are best to use ). 
2) fork keras and add cynfrence as a backend with all the learning capabilities turned off. 



# Supported layers 
I'm sowly adding all generic keras layers. If you need a specific layer feel free to PM me and I'll work on it in advance. you can also just try to create it your self - its super easy given the layers that already exist.

I only list the layers which are selected for dev:

* Core layers
- [ ] Dense
- [ ] Activation
- [ ] Dropout
- [ ] Dense
- [ ] Flatten
- [ ] Input
- [ ] Reshape

* Convolutional layers
- [ ] Conv1D
- [x] Conv2D
- [ ] SeperableConv1D
- [x] SeperableConv2D
- [ ] Conv2DTranspose
- [ ] Conv3D
- [ ] Cropping1D
- [ ] Cropping2D
- [ ] Cropping3D
- [ ] UpSampling1D
- [ ] UpSampling2D
- [ ] UpSampling3D
- [ ] ZeroPadding1D
- [ ] ZeroPadding2D
- [ ] ZeroPadding3D 

* Pooling layers
- [ ] MaxPooling1D
- [ ] MaxPooling2D
- [ ] MaxPooling3D
- [ ] AveragePooling1D
- [ ] AveragePooling2D
- [ ] AveragePooling3D
- [ ] GlobalMaxPooling1D
- [ ] GlobalMaxPooling2D
- [ ] GlobalMaxPooling3D
- [ ] GlobalAveragePooling1D
- [ ] GlobalAveragePooling2D
- [ ] GlobalAveragePooling3D 

* Reccurent Layers
- [ ] RNN
- [ ] SimpleRNN
- [ ] GRU
- [ ] LSTM

* Merge Layers
- [ ] Add
- [ ] Subtract
- [ ] Mutiply
- [ ] Average
- [ ] Average
- [ ] Maximum
- [ ] Concatenate 

* Noemalization Layers
- [ ] BatchNormalization

* Activations
- [ ] softmax
- [ ] relu
- [ ] tanh
- [ ] sigmoid
- [x] None / linear 












