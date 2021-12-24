import os
import time
import ctypes
from ctypes import Structure

import numpy as np
import torch


SO_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build/libconv2d.so")
cpplib = ctypes.cdll.LoadLibrary(SO_FILENAME)


def conv2d_torch(x, weight, stride=1, padding=0, dilation=1, groups=1):
    x = torch.from_numpy(x)
    weight = torch.from_numpy(weight)
    y = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, 
                                   padding=padding, dilation=dilation, groups=groups) 
    return y.numpy()


class Tensor4fCtypes(Structure):
    _fields_ = [('data', ctypes.POINTER(ctypes.c_float)),
                ('dims', ctypes.c_int * 4)]
                

def compute_output_size(input_size, filter_size, stride=1,  padding_begin=0, padding_end=0, dilation=1):
    filter_extent = dilation * (filter_size - 1) + 1
    output_size = (input_size - filter_extent + padding_begin + padding_end) // stride + 1
    return output_size


def conv2d_with_stride(x, weight, stride_h=1, stride_w=1, version='v1'):
    batch_size, _, input_height, input_width = x.shape
    out_channels, _, filter_height, filter_width = weight.shape

    output_height = (input_height - filter_height) // stride_h + 1
    output_width = (input_width - filter_width) // stride_w + 1
    y = np.empty((batch_size, out_channels, output_height, output_width), np.float32)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    x_ctypes = Tensor4fCtypes()
    x_ctypes.data = x.ctypes.data_as(c_float_p)
    x_ctypes.dims = (ctypes.c_int * len(x.shape))(*x.shape)
    y_ctypes = Tensor4fCtypes()
    y_ctypes.data = y.ctypes.data_as(c_float_p)
    y_ctypes.dims = (ctypes.c_int * len(y.shape))(*y.shape)
    w_ctypes = Tensor4fCtypes()
    w_ctypes.data = weight.ctypes.data_as(c_float_p)
    w_ctypes.dims = (ctypes.c_int * len(weight.shape))(*weight.shape)

    cpplib.conv2d_with_stride.argtypes = [ctypes.POINTER(Tensor4fCtypes),
                     ctypes.POINTER(Tensor4fCtypes),
                     ctypes.POINTER(Tensor4fCtypes),
                     ctypes.c_int, ctypes.c_int]
    cpplib.conv2d_with_stride(x_ctypes, w_ctypes,  y_ctypes, stride_h, stride_w)
    return y


def conv2d_with_stride_padding(x, weight, stride_h=1, stride_w=1, 
                               padding_h_begin=0, padding_h_end=0, 
                               padding_w_begin=0, padding_w_end=0):
    batch_size, _, input_height, input_width = x.shape
    out_channels, _, filter_height, filter_width = weight.shape

    output_height = (input_height - filter_height + padding_h_begin + padding_h_end) // stride_h + 1
    output_width = (input_width - filter_width + padding_w_begin + padding_w_end) // stride_w + 1
    y = np.empty((batch_size, out_channels, output_height, output_width), np.float32)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    x_ctypes = Tensor4fCtypes()
    x_ctypes.data = x.ctypes.data_as(c_float_p)
    x_ctypes.dims = (ctypes.c_int * len(x.shape))(*x.shape)
    y_ctypes = Tensor4fCtypes()
    y_ctypes.data = y.ctypes.data_as(c_float_p)
    y_ctypes.dims = (ctypes.c_int * len(y.shape))(*y.shape)
    w_ctypes = Tensor4fCtypes()
    w_ctypes.data = weight.ctypes.data_as(c_float_p)
    w_ctypes.dims = (ctypes.c_int * len(weight.shape))(*weight.shape)

    cpplib.conv2d_with_stride_padding.argtypes = [ctypes.POINTER(Tensor4fCtypes),
                                                  ctypes.POINTER(Tensor4fCtypes),
                                                  ctypes.POINTER(Tensor4fCtypes),
                                                  ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_int, ctypes.c_int]
    cpplib.conv2d_with_stride_padding(x_ctypes, w_ctypes,  y_ctypes, stride_h, stride_w, 
                                      padding_h_begin, padding_h_end, padding_w_begin, padding_w_end)
    return y


def conv2d_with_stride_padding_dilation(x, weight, stride_h=1, stride_w=1, 
                                        padding_h_begin=0, padding_h_end=0, 
                                        padding_w_begin=0, padding_w_end=0,
                                        dilation_h=1, dilation_w=1):
    batch_size, _, input_height, input_width = x.shape
    out_channels, _, filter_height, filter_width = weight.shape

    filter_height_valid = dilation_h * (filter_height - 1) + 1
    filter_width_valid = dilation_w * (filter_width - 1) + 1
    output_height = (input_height - filter_height_valid + padding_h_begin + padding_h_end) // stride_h + 1
    output_width = (input_width - filter_width_valid + padding_w_begin + padding_w_end) // stride_w + 1
    y = np.empty((batch_size, out_channels, output_height, output_width), np.float32)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    x_ctypes = Tensor4fCtypes()
    x_ctypes.data = x.ctypes.data_as(c_float_p)
    x_ctypes.dims = (ctypes.c_int * len(x.shape))(*x.shape)
    y_ctypes = Tensor4fCtypes()
    y_ctypes.data = y.ctypes.data_as(c_float_p)
    y_ctypes.dims = (ctypes.c_int * len(y.shape))(*y.shape)
    w_ctypes = Tensor4fCtypes()
    w_ctypes.data = weight.ctypes.data_as(c_float_p)
    w_ctypes.dims = (ctypes.c_int * len(weight.shape))(*weight.shape)

    cpplib.conv2d_with_stride_padding_dilation.argtypes = [
        ctypes.POINTER(Tensor4fCtypes),
        ctypes.POINTER(Tensor4fCtypes),
        ctypes.POINTER(Tensor4fCtypes),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int]
    cpplib.conv2d_with_stride_padding_dilation(
        x_ctypes, w_ctypes,  y_ctypes, stride_h, stride_w, 
        padding_h_begin, padding_h_end, padding_w_begin, padding_w_end,
        dilation_h, dilation_w)
    return y


def _conv2d_with_im2col(
        x, weight, stride_h=1, stride_w=1, 
        padding_h_begin=0, padding_h_end=0, 
        padding_w_begin=0, padding_w_end=0,
        dilation_h=1, dilation_w=1, groups=1):
    batch_size, in_channels, input_height, input_width = x.shape
    out_channels, _, filter_height, filter_width = weight.shape
    output_height = compute_output_size(input_height, filter_height, stride=stride_h,  
                                        padding_begin=padding_h_begin, padding_end=padding_h_end, dilation=dilation_h)
    output_width= compute_output_size(input_width, filter_width, stride=stride_w,  
                                      padding_begin=padding_w_begin, padding_end=padding_w_end, dilation=dilation_w)

    output = np.empty((batch_size, out_channels, output_height, output_width), np.float32)

    assert x.shape[1] // groups == weight.shape[1]
    assert padding_h_begin == padding_h_end
    assert padding_w_begin == padding_w_end
    assert in_channels % groups == 0
    assert out_channels % groups == 0
    
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    c_float_p = ctypes.POINTER(ctypes.c_float)
    cpplib.im2col_cpu.argtypes = [
        c_float_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, 
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        c_float_p]

    data_col = np.empty((in_channels_per_group * filter_height * filter_width, output_height * output_width), dtype=np.float32)
    for batch_index in range(batch_size):
        for group_index in range(groups):
            input_part = x[batch_index, group_index * in_channels_per_group: (group_index + 1) * in_channels_per_group, ...]
            filter_part = weight[group_index * out_channels_per_group: (group_index + 1) * out_channels_per_group, ...].reshape(out_channels_per_group, -1)
            input_part_ptr = input_part.ctypes.data_as(c_float_p)
            data_col_ptr = data_col.ctypes.data_as(c_float_p)
            cpplib.im2col_cpu(input_part_ptr, in_channels_per_group,
                              input_height, input_width, filter_height, filter_width,
                              padding_h_begin, padding_w_begin, stride_h, stride_w, dilation_h, dilation_w, data_col_ptr)
            # filter_ptr: (out_channels_per_group, in_channels_per_group * filter_height * filter_width)
            # data_col: (in_channels_per_group * filter_height * filter_width, output_height * output_width)
            # output_ptr: (out_channels_per_group, output_height * output_width)
            output[batch_index, group_index * out_channels_per_group: (group_index + 1) * out_channels_per_group, ...] = \
                np.dot(filter_part, data_col).reshape(out_channels_per_group, output_height, output_width)
    return output


def conv2d_with_stride_padding_dilation_groups(
        x, weight, stride_h=1, stride_w=1, 
        padding_h_begin=0, padding_h_end=0, 
        padding_w_begin=0, padding_w_end=0,
        dilation_h=1, dilation_w=1, groups=1,
        version='raw'):
    if version == 'raw':
        func = cpplib.conv2d_with_stride_padding_dilation_groups
    elif version == 'im2col':
        func = cpplib.conv2d_with_stride_padding_dilation_groups_im2col
    elif version == 'im2col_v2':
        return _conv2d_with_im2col(
            x, weight, stride_h=stride_h, stride_w=stride_w, 
            padding_h_begin=padding_h_begin, padding_h_end=padding_h_end, 
            padding_w_begin=padding_w_begin, padding_w_end=padding_w_end,
            dilation_h=dilation_h, dilation_w=dilation_w, groups=groups)

    batch_size, _, input_height, input_width = x.shape
    out_channels, _, filter_height, filter_width = weight.shape

    filter_height_valid = dilation_h * (filter_height - 1) + 1
    filter_width_valid = dilation_w * (filter_width - 1) + 1
    output_height = (input_height - filter_height_valid + padding_h_begin + padding_h_end) // stride_h + 1
    output_width = (input_width - filter_width_valid + padding_w_begin + padding_w_end) // stride_w + 1
    y = np.empty((batch_size, out_channels, output_height, output_width), np.float32)

    c_float_p = ctypes.POINTER(ctypes.c_float)
    x_ctypes = Tensor4fCtypes()
    x_ctypes.data = x.ctypes.data_as(c_float_p)
    x_ctypes.dims = (ctypes.c_int * len(x.shape))(*x.shape)
    y_ctypes = Tensor4fCtypes()
    y_ctypes.data = y.ctypes.data_as(c_float_p)
    y_ctypes.dims = (ctypes.c_int * len(y.shape))(*y.shape)
    w_ctypes = Tensor4fCtypes()
    w_ctypes.data = weight.ctypes.data_as(c_float_p)
    w_ctypes.dims = (ctypes.c_int * len(weight.shape))(*weight.shape)

    func.argtypes = [
        ctypes.POINTER(Tensor4fCtypes),
        ctypes.POINTER(Tensor4fCtypes),
        ctypes.POINTER(Tensor4fCtypes),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, 
        ctypes.c_int]
    func(x_ctypes, w_ctypes,  y_ctypes, stride_h, stride_w, 
         padding_h_begin, padding_h_end, padding_w_begin, padding_w_end,
         dilation_h, dilation_w, groups)
    return y


if __name__ == '__main__':
    batch_size = np.random.choice(np.arange(1, 9))
    input_height = np.random.choice(np.arange(16, 33))
    input_width = np.random.choice(np.arange(16, 33))
    filter_height = np.random.choice(np.arange(2, 5))
    filter_width = np.random.choice(np.arange(2, 5))
    in_channels = np.random.choice(np.arange(1, 12))
    out_channels = np.random.choice(np.arange(1, 128))
    x = np.random.randn(batch_size, in_channels, input_height, input_width).astype(np.float32)
    weight = np.random.randn(out_channels, in_channels, filter_height, filter_width).astype(np.float32)

    stride_h = np.random.choice([1, 2, 3])
    stride_w = np.random.choice([1, 2, 3])
    padding_h = np.random.choice([1, 2, 3])
    padding_w = np.random.choice([1, 2, 3])
    dilation_h = np.random.choice([1, 2, 3])
    dilation_w = np.random.choice([1, 2, 3])

    start_time = time.time()
    y_np = conv2d_torch(x, weight, stride=(stride_h, stride_w))
    print(time.time() - start_time)
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride(x, weight, stride_h=stride_h, stride_w=stride_w)
    print(time.time() - start_time)
    print(np.allclose(y_np, y_cpp, atol=1e-3))
    print(np.mean(np.abs(y_np - y_cpp)))
    print("===================")

    start_time = time.time()
    y_np = conv2d_torch(x, weight, stride=(stride_h, stride_w), padding=(padding_h, padding_w))
    print(time.time() - start_time)
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride_padding(x, weight, stride_h=stride_h, stride_w=stride_w, 
                                       padding_h_begin=padding_h, padding_h_end=padding_h, 
                                       padding_w_begin=padding_w, padding_w_end=padding_w)
    print(time.time() - start_time)
    print(np.allclose(y_np, y_cpp, atol=1e-3))
    print(np.mean(np.abs(y_np - y_cpp)))
    print("===================")

    start_time = time.time()
    y_np = conv2d_torch(x, weight, stride=(stride_h, stride_w), 
                        padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w))
    print(time.time() - start_time)
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride_padding_dilation(x, weight, stride_h=stride_h, stride_w=stride_w, 
                                                padding_h_begin=padding_h, padding_h_end=padding_h, 
                                                padding_w_begin=padding_w, padding_w_end=padding_w,
                                                dilation_h=dilation_h, dilation_w=dilation_w)
    print(time.time() - start_time)
    print(np.allclose(y_np, y_cpp, atol=1e-3))
    print(np.mean(np.abs(y_np - y_cpp)))
    print("==================")


    batch_size = np.random.choice(np.arange(1, 9))
    input_height = np.random.choice(np.arange(16, 33))
    input_width = np.random.choice(np.arange(16, 33))
    filter_height = np.random.choice(np.arange(2, 5))
    filter_width = np.random.choice(np.arange(2, 5))
    groups = np.random.choice([1, 2, 4, 8])
    in_channels, out_channels = 32, 128
    in_channels_f = in_channels // groups
    x = np.random.randn(batch_size, in_channels, input_height, input_width).astype(np.float32)
    weight = np.random.randn(out_channels, in_channels_f, filter_height, filter_width).astype(np.float32)

    stride_h = np.random.choice([1, 2, 3])
    stride_w = np.random.choice([1, 2, 3])
    padding_h = np.random.choice([1, 2, 3])
    padding_w = np.random.choice([1, 2, 3])
    dilation_h = np.random.choice([1, 2, 3])
    dilation_w = np.random.choice([1, 2, 3])
    
    start_time = time.time()
    y_np = conv2d_torch(x, weight, stride=(stride_h, stride_w), 
                        padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w), groups=groups)
    print(time.time() - start_time)
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride_padding_dilation_groups(
        x, weight, stride_h=stride_h, stride_w=stride_w, 
        padding_h_begin=padding_h, padding_h_end=padding_h, 
        padding_w_begin=padding_w, padding_w_end=padding_w,
        dilation_h=dilation_h, dilation_w=dilation_w,
        groups=groups)
    print(time.time() - start_time)
    print(np.allclose(y_np, y_cpp, atol=1e-3))
    print(np.mean(np.abs(y_np - y_cpp)))
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride_padding_dilation_groups(
        x, weight, stride_h=stride_h, stride_w=stride_w, 
        padding_h_begin=padding_h, padding_h_end=padding_h, 
        padding_w_begin=padding_w, padding_w_end=padding_w,
        dilation_h=dilation_h, dilation_w=dilation_w,
        groups=groups, version='im2col')
    print(time.time() - start_time)
    print(np.allclose(y_np, y_cpp, atol=1e-3))
    print(np.mean(np.abs(y_np - y_cpp)))
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride_padding_dilation_groups(
        x, weight, stride_h=stride_h, stride_w=stride_w, 
        padding_h_begin=padding_h, padding_h_end=padding_h, 
        padding_w_begin=padding_w, padding_w_end=padding_w,
        dilation_h=dilation_h, dilation_w=dilation_w,
        groups=groups, version='im2col_v2')
    print(time.time() - start_time)
    print(np.allclose(y_np, y_cpp, atol=1e-3))
    print(np.mean(np.abs(y_np - y_cpp)))
