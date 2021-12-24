# Overview
Implement convolution layer in CNN step by step, regardless of efficiency.

Currently including:
- conv2d_with_stride
- conv2d_with_stride_padding
- conv2d_with_stride_padding_dilation
- conv2d_with_stride_padding_dilation_groups
    - version='raw': plain implementation
    - version='im2col': implementation using im2col borrowed from Caffe.

We compare the results of our implementions with Pytorch. 

## Dependancy
- NumPy
- PyTorch

## Usage
```
mkdir -p build
cmake ..
make
cd ..
python test.py
```
