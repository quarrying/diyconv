# Overview
Implement convolution layer in CNN step by step for ease of understanding, regardless of efficiency.

Currently including:
- conv2d_with_stride
- conv2d_with_stride_padding
- conv2d_with_stride_padding_dilation
- conv2d_with_stride_padding_dilation_groups
    - `mode='plain'`: plain implementation.
    - `mode='im2col'`: implementation using `im2col` borrowed from Caffe and `matmul` implemented by ourselves.
    - `mode='im2col_v2'`: implementation using `im2col` borrowed from Caffe and `np.dot`.

We also compare the results of our implementions with Pytorch's. 

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
