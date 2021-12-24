
import time

import numpy as np
import torch
from diy_conv import conv2d_with_stride
from diy_conv import conv2d_with_stride_padding
from diy_conv import conv2d_with_stride_padding_dilation
from diy_conv import conv2d_with_stride_padding_dilation_groups


def conv2d_torch(x, weight, stride=1, padding=0, dilation=1, groups=1):
    x = torch.from_numpy(x)
    weight = torch.from_numpy(weight)
    y = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, 
                                   padding=padding, dilation=dilation, groups=groups) 
    return y.numpy()


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
    y_torch = conv2d_torch(x, weight, stride=(stride_h, stride_w))
    print(time.time() - start_time)
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride(x, weight, stride_h=stride_h, stride_w=stride_w)
    print(time.time() - start_time)
    print('Close: {}, MAE: {}'.format(np.allclose(y_torch, y_cpp, atol=1e-3), np.mean(np.abs(y_torch - y_cpp))))
    print("===================")

    start_time = time.time()
    y_torch = conv2d_torch(x, weight, stride=(stride_h, stride_w), padding=(padding_h, padding_w))
    print(time.time() - start_time)
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride_padding(x, weight, stride_h=stride_h, stride_w=stride_w, 
                                       padding_h_begin=padding_h, padding_h_end=padding_h, 
                                       padding_w_begin=padding_w, padding_w_end=padding_w)
    print(time.time() - start_time)
    print('Close: {}, MAE: {}'.format(np.allclose(y_torch, y_cpp, atol=1e-3), np.mean(np.abs(y_torch - y_cpp))))
    print("===================")

    start_time = time.time()
    y_torch = conv2d_torch(x, weight, stride=(stride_h, stride_w), 
                        padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w))
    print(time.time() - start_time)
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride_padding_dilation(x, weight, stride_h=stride_h, stride_w=stride_w, 
                                                padding_h_begin=padding_h, padding_h_end=padding_h, 
                                                padding_w_begin=padding_w, padding_w_end=padding_w,
                                                dilation_h=dilation_h, dilation_w=dilation_w)
    print(time.time() - start_time)
    print('Close: {}, MAE: {}'.format(np.allclose(y_torch, y_cpp, atol=1e-3), np.mean(np.abs(y_torch - y_cpp))))
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
    y_torch = conv2d_torch(x, weight, stride=(stride_h, stride_w), 
                        padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w), groups=groups)
    print(time.time() - start_time)
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride_padding_dilation_groups(
        x, weight, stride_h=stride_h, stride_w=stride_w, 
        padding_h_begin=padding_h, padding_h_end=padding_h, 
        padding_w_begin=padding_w, padding_w_end=padding_w,
        dilation_h=dilation_h, dilation_w=dilation_w,
        groups=groups, mode='plain')
    print(time.time() - start_time)
    print('Close: {}, MAE: {}'.format(np.allclose(y_torch, y_cpp, atol=1e-3), np.mean(np.abs(y_torch - y_cpp))))
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride_padding_dilation_groups(
        x, weight, stride_h=stride_h, stride_w=stride_w, 
        padding_h_begin=padding_h, padding_h_end=padding_h, 
        padding_w_begin=padding_w, padding_w_end=padding_w,
        dilation_h=dilation_h, dilation_w=dilation_w,
        groups=groups, mode='im2col')
    print(time.time() - start_time)
    print('Close: {}, MAE: {}'.format(np.allclose(y_torch, y_cpp, atol=1e-3), np.mean(np.abs(y_torch - y_cpp))))
    print("-------------------")

    start_time = time.time()
    y_cpp = conv2d_with_stride_padding_dilation_groups(
        x, weight, stride_h=stride_h, stride_w=stride_w, 
        padding_h_begin=padding_h, padding_h_end=padding_h, 
        padding_w_begin=padding_w, padding_w_end=padding_w,
        dilation_h=dilation_h, dilation_w=dilation_w,
        groups=groups, mode='im2col_v2')
    print(time.time() - start_time)
    print('Close: {}, MAE: {}'.format(np.allclose(y_torch, y_cpp, atol=1e-3), np.mean(np.abs(y_torch - y_cpp))))
