#include <assert.h>
#include <stdio.h>

typedef struct Tensor
{
    float* data;
    int    dims[4];
} Tensor4f;

extern "C"
{
    void conv2d_with_stride(
        const Tensor4f* input, const Tensor4f* filter, Tensor4f* output, 
        int stride_h, int stride_w);

    void conv2d_with_stride_padding(
        const Tensor4f* input, const Tensor4f* filter, Tensor4f* output, 
        int stride_h, int stride_w, 
        int padding_h_begin, int padding_h_end, 
        int padding_w_begin, int padding_w_end);

    void conv2d_with_stride_padding_dilation(
        const Tensor4f* input, const Tensor4f* filter, Tensor4f* output, 
        int stride_h, int stride_w, 
        int padding_h_begin, int padding_h_end, 
        int padding_w_begin, int padding_w_end,
        int dilation_h, int dilation_w);

    void conv2d_with_stride_padding_dilation_groups(
        const Tensor4f* input, const Tensor4f* filter, Tensor4f* output, 
        int stride_h, int stride_w, 
        int padding_h_begin, int padding_h_end, 
        int padding_w_begin, int padding_w_end,
        int dilation_h, int dilation_w,
        int groups);
        
    void conv2d_with_stride_padding_dilation_groups_by_im2col(
        const Tensor4f* input, const Tensor4f* filter, Tensor4f* output, 
        int stride_h, int stride_w, 
        int padding_h_begin, int padding_h_end, 
        int padding_w_begin, int padding_w_end,
        int dilation_h, int dilation_w,
        int groups);

    void im2col_cpu(const float* data_im, const int channels,
                    const int height, const int width, const int kernel_h, const int kernel_w,
                    const int pad_h, const int pad_w, const int stride_h,
                    const int stride_w, const int dilation_h, const int dilation_w,
                    float* data_col);
}

void printTensor(const Tensor4f* tensor)
{
    printf("(%d, %d, %d, %d)\n", tensor->dims[0], tensor->dims[1], tensor->dims[2], tensor->dims[3]);
}


// input:   (batch_size, in_channels, input_height, input_width)
// filter:  (out_channels, in_channels, filter_height, filter_width)
// output:  (batch_size, out_channels, output_height, output_width)
void conv2d_with_stride(
    const Tensor4f* input, const Tensor4f* filter, Tensor4f* output, 
    int stride_h, int stride_w)
{
    assert (input->dims[0] == output->dims[0]);
    assert (input->dims[1] == filter->dims[1]);
    assert (filter->dims[0] == output->dims[1]);
    
    int batch_size = input->dims[0];
    int in_channels = input->dims[1];
    int input_height = input->dims[2];
    int input_width = input->dims[3];
    
    int out_channels = filter->dims[0];
    int filter_height = filter->dims[2];
    int filter_width = filter->dims[3];
    
    int output_height = output->dims[2];
    int output_width = output->dims[3];
    
    int input_numel = in_channels * input_height * input_width;
    int output_numel = out_channels * output_height * output_width;
    int filter_numel = in_channels * filter_height * filter_width;
    for (int batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        int input_offset = batch_index * input_numel;
        int output_offset = batch_index * output_numel;
        const float* input_ptr = input->data + input_offset;
        float* output_ptr = output->data + output_offset;
        for (int output_c_index = 0; output_c_index < out_channels; ++output_c_index)
        {
            int filter_offset = output_c_index * filter_numel;
            const float* filter_ptr = filter->data + filter_offset;
            for (int output_h_index = 0; output_h_index < output_height; ++output_h_index)
            {
                for (int output_w_index = 0; output_w_index < output_width; ++output_w_index)
                {
                    float val = 0.0f;
                    for (int filter_c_index = 0; filter_c_index < in_channels; ++filter_c_index)
                    {
                        for (int filter_h_index = 0; filter_h_index < filter_height; ++filter_h_index)
                        {
                            for (int filter_w_index = 0; filter_w_index < filter_width; ++filter_w_index)
                            {
                                int input_idx = 
                                    (filter_w_index + output_w_index * stride_w) +
                                    (filter_h_index + output_h_index * stride_h) * input_width + 
                                    filter_c_index * input_width * input_height;
                                int filter_idx = 
                                    filter_w_index + 
                                    filter_h_index * filter_width + 
                                    filter_c_index * filter_width * filter_height;
                                val += input_ptr[input_idx] * filter_ptr[filter_idx];
                            }
                        }
                    }
                    int output_idx =
                        output_w_index +
                        output_h_index * output_width +
                        output_c_index * output_width * output_height;
                    output_ptr[output_idx] = val;
                }
            }
        }
    }
}


// input:   (batch_size, in_channels, input_height, input_width)
// filter:  (out_channels, in_channels, filter_height, filter_width)
// output:  (batch_size, out_channels, output_height, output_width)
void conv2d_with_stride_padding(
    const Tensor4f* input, const Tensor4f* filter, Tensor4f* output, 
    int stride_h, int stride_w, 
    int padding_h_begin, int padding_h_end, 
    int padding_w_begin, int padding_w_end)
{
    assert (input->dims[0] == output->dims[0]);
    assert (input->dims[1] == filter->dims[1]);
    assert (filter->dims[0] == output->dims[1]);
    
    int batch_size = input->dims[0];
    int in_channels = input->dims[1];
    int input_height = input->dims[2];
    int input_width = input->dims[3];
    
    int out_channels = filter->dims[0];
    int filter_height = filter->dims[2];
    int filter_width = filter->dims[3];
    
    int output_height = output->dims[2];
    int output_width = output->dims[3];
    
    int input_numel = in_channels * input_height * input_width;
    int output_numel = out_channels * output_height * output_width;
    int filter_numel = in_channels * filter_height * filter_width;
    for (int batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        int input_offset = batch_index * input_numel;
        int output_offset = batch_index * output_numel;
        const float* input_ptr = input->data + input_offset;
        float* output_ptr = output->data + output_offset;
        for (int output_c_index = 0; output_c_index < out_channels; ++output_c_index)
        {
            int filter_offset = output_c_index * filter_numel;
            const float* filter_ptr = filter->data + filter_offset;
            for (int output_h_index = 0; output_h_index < output_height; ++output_h_index)
            {
                for (int output_w_index = 0; output_w_index < output_width; ++output_w_index)
                {
                    float val = 0.0f;
                    for (int filter_c_index = 0; filter_c_index < in_channels; ++filter_c_index)
                    {
                        for (int filter_h_index = 0; filter_h_index < filter_height; ++filter_h_index)
                        {
                            int input_h_index = filter_h_index + output_h_index * stride_h - padding_h_begin;
                            if ((input_h_index < 0) || (input_h_index >= input_height))
                            {
                                continue;
                            }
                            for (int filter_w_index = 0; filter_w_index < filter_width; ++filter_w_index)
                            {
                                int input_w_index = filter_w_index + output_w_index * stride_w - padding_w_begin;
                                if ((input_w_index < 0) || (input_w_index >= input_width))
                                {
                                    continue;
                                }
                                int input_idx = 
                                    input_w_index +
                                    input_h_index * input_width + 
                                    filter_c_index * input_width * input_height;
                                int filter_idx = 
                                    filter_w_index + 
                                    filter_h_index * filter_width + 
                                    filter_c_index * filter_width * filter_height;
                                val += input_ptr[input_idx] * filter_ptr[filter_idx];
                            }
                        }
                    }
                    int output_idx =
                        output_w_index +
                        output_h_index * output_width +
                        output_c_index * output_width * output_height;
                    output_ptr[output_idx] = val;
                }
            }
        }
    }
}


// input:   (batch_size, in_channels, input_height, input_width)
// filter:  (out_channels, in_channels, filter_height, filter_width)
// output:  (batch_size, out_channels, output_height, output_width)
void conv2d_with_stride_padding_dilation(
    const Tensor4f* input, const Tensor4f* filter, Tensor4f* output, 
    int stride_h, int stride_w, 
    int padding_h_begin, int padding_h_end, 
    int padding_w_begin, int padding_w_end,
    int dilation_h, int dilation_w)
{
    assert (input->dims[0] == output->dims[0]);
    assert (input->dims[1] == filter->dims[1]);
    assert (filter->dims[0] == output->dims[1]);
    
    int batch_size = input->dims[0];
    int in_channels = input->dims[1];
    int input_height = input->dims[2];
    int input_width = input->dims[3];
    
    int out_channels = filter->dims[0];
    int filter_height = filter->dims[2];
    int filter_width = filter->dims[3];
    
    int output_height = output->dims[2];
    int output_width = output->dims[3];
    
    int input_numel = in_channels * input_height * input_width;
    int output_numel = out_channels * output_height * output_width;
    int filter_numel = in_channels * filter_height * filter_width;
    for (int batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        int input_offset = batch_index * input_numel;
        int output_offset = batch_index * output_numel;
        const float* input_ptr = input->data + input_offset;
        float* output_ptr = output->data + output_offset;
        for (int output_c_index = 0; output_c_index < out_channels; ++output_c_index)
        {
            int filter_offset = output_c_index * filter_numel;
            const float* filter_ptr = filter->data + filter_offset;
            for (int output_h_index = 0; output_h_index < output_height; ++output_h_index)
            {
                for (int output_w_index = 0; output_w_index < output_width; ++output_w_index)
                {
                    float val = 0.0f;
                    for (int filter_c_index = 0; filter_c_index < in_channels; ++filter_c_index)
                    {
                        for (int filter_h_index = 0; filter_h_index < filter_height; ++filter_h_index)
                        {
                            int input_h_index = filter_h_index * dilation_h + output_h_index * stride_h - padding_h_begin;
                            if ((input_h_index < 0) || (input_h_index >= input_height))
                            {
                                continue;
                            }
                            for (int filter_w_index = 0; filter_w_index < filter_width; ++filter_w_index)
                            {
                                int input_w_index = filter_w_index * dilation_w + output_w_index * stride_w - padding_w_begin;
                                if ((input_w_index < 0) || (input_w_index >= input_width))
                                {
                                    continue;
                                }
                                int input_idx = 
                                    input_w_index +
                                    input_h_index * input_width + 
                                    filter_c_index * input_width * input_height;
                                int filter_idx = 
                                    filter_w_index + 
                                    filter_h_index * filter_width + 
                                    filter_c_index * filter_width * filter_height;
                                val += input_ptr[input_idx] * filter_ptr[filter_idx];
                            }
                        }
                    }
                    int output_idx =
                        output_w_index +
                        output_h_index * output_width +
                        output_c_index * output_width * output_height;
                    output_ptr[output_idx] = val;
                }
            }
        }
    }
}

// input:   (batch_size, in_channels, input_height, input_width)
// filter:  (out_channels, in_channels // groups, filter_height, filter_width)
// output:  (batch_size, out_channels, output_height, output_width)
void conv2d_with_stride_padding_dilation_groups(
    const Tensor4f* input, const Tensor4f* filter, Tensor4f* output, 
    int stride_h, int stride_w, 
    int padding_h_begin, int padding_h_end, 
    int padding_w_begin, int padding_w_end,
    int dilation_h, int dilation_w,
    int groups)
{
    assert (input->dims[0] == output->dims[0]);
    assert (input->dims[1] / groups == filter->dims[1]);
    assert (filter->dims[0] == output->dims[1]);
    
    int batch_size = input->dims[0];
    int in_channels = input->dims[1];
    int input_height = input->dims[2];
    int input_width = input->dims[3];
    
    int out_channels = filter->dims[0];
    int filter_height = filter->dims[2];
    int filter_width = filter->dims[3];
    
    int output_height = output->dims[2];
    int output_width = output->dims[3];
    
    assert (in_channels % groups == 0);
    assert (out_channels % groups == 0);
    
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    int input_numel = in_channels * input_height * input_width;
    int output_numel = out_channels * output_height * output_width;
    int filter_numel = in_channels_per_group * filter_height * filter_width;
    for (int batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        for (int group_index = 0; group_index < groups; ++group_index)
        {
            int input_offset = batch_index * input_numel + group_index * in_channels_per_group * input_height * input_width;
            int output_offset = batch_index * output_numel + group_index * out_channels_per_group  * output_height * output_width;
            const float* input_ptr = input->data + input_offset;
            float* output_ptr = output->data + output_offset;
            for (int output_c_index = 0; output_c_index < out_channels_per_group; ++output_c_index)
            {
                int filter_offset = (output_c_index + group_index * out_channels_per_group) * filter_numel;
                const float* filter_ptr = filter->data + filter_offset;
                for (int output_h_index = 0; output_h_index < output_height; ++output_h_index)
                {
                    for (int output_w_index = 0; output_w_index < output_width; ++output_w_index)
                    {
                        float val = 0.0f;
                        for (int filter_c_index = 0; filter_c_index < in_channels_per_group; ++filter_c_index)
                        {
                            for (int filter_h_index = 0; filter_h_index < filter_height; ++filter_h_index)
                            {
                                int input_h_index = filter_h_index * dilation_h + output_h_index * stride_h - padding_h_begin;
                                if ((input_h_index < 0) || (input_h_index >= input_height))
                                {
                                    continue;
                                }
                                for (int filter_w_index = 0; filter_w_index < filter_width; ++filter_w_index)
                                {
                                    int input_w_index = filter_w_index * dilation_w + output_w_index * stride_w - padding_w_begin;
                                    if ((input_w_index < 0) || (input_w_index >= input_width))
                                    {
                                        continue;
                                    }
                                    int input_idx = 
                                        input_w_index +
                                        input_h_index * input_width + 
                                        filter_c_index * input_width * input_height;
                                    int filter_idx = 
                                        filter_w_index + 
                                        filter_h_index * filter_width + 
                                        filter_c_index * filter_width * filter_height;
                                    val += input_ptr[input_idx] * filter_ptr[filter_idx];
                                }
                            }
                        }
                        int output_idx =
                            output_w_index +
                            output_h_index * output_width +
                            output_c_index * output_width * output_height;
                        output_ptr[output_idx] = val;
                    } // end for (int output_w_index = 0; output_w_index < output_width; ++output_w_index)
                } // end for (int output_h_index = 0; output_h_index < output_height; ++output_h_index)
            } // end for (int output_c_index = 0; output_c_index < out_channels_per_group; ++output_c_index)
        } // end for (int group_index = 0; group_index < groups; ++group_index)
    } // end for (int batch_index = 0; batch_index < batch_size; ++batch_index)
}


// Borrowed from Caffe
// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}


// Borrowed from Caffe
template <typename Dtype>
void im2col_cpu_(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}


void im2col_cpu(const float* data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
                float* data_col)
{
    im2col_cpu_(data_im, channels, height, width, kernel_h, kernel_w,
                pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, data_col);
}


// Simplest matrix multiplication implementation
void matmul(const float* a, const float* b, float* c, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            float sum = 0;
            for (int h = 0; h < n; ++h)
            {
                sum += a[i * n + h] * b[h * k + j];
            }
            c[i * k + j] = sum;
        }
    }
}


// input:   (batch_size, in_channels, input_height, input_width)
// filter:  (out_channels, in_channels // groups, filter_height, filter_width)
// output:  (batch_size, out_channels, output_height, output_width)
void conv2d_with_stride_padding_dilation_groups_by_im2col(
    const Tensor4f* input, const Tensor4f* filter, Tensor4f* output, 
    int stride_h, int stride_w, 
    int padding_h_begin, int padding_h_end, 
    int padding_w_begin, int padding_w_end,
    int dilation_h, int dilation_w,
    int groups)
{
    assert (input->dims[0] == output->dims[0]);
    assert (input->dims[1] / groups == filter->dims[1]);
    assert (filter->dims[0] == output->dims[1]);
    assert (padding_h_begin == padding_h_end);
    assert (padding_w_begin == padding_w_end);

    int batch_size = input->dims[0];
    int in_channels = input->dims[1];
    int input_height = input->dims[2];
    int input_width = input->dims[3];
    
    int out_channels = filter->dims[0];
    int filter_height = filter->dims[2];
    int filter_width = filter->dims[3];
    
    int output_height = output->dims[2];
    int output_width = output->dims[3];
    
    assert (in_channels % groups == 0);
    assert (out_channels % groups == 0);
    
    int in_channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    int input_numel = in_channels * input_height * input_width;
    int output_numel = out_channels * output_height * output_width;
    int filter_numel = in_channels_per_group * filter_height * filter_width;

    float* data_col = new float[in_channels_per_group * filter_height * filter_width * output_height * output_width];
    for (int batch_index = 0; batch_index < batch_size; ++batch_index)
    {
        for (int group_index = 0; group_index < groups; ++group_index)
        {
            int input_offset = batch_index * input_numel + group_index * in_channels_per_group * input_height * input_width;
            int filter_offset = group_index * out_channels_per_group * filter_numel;
            int output_offset = batch_index * output_numel + group_index * out_channels_per_group * output_height * output_width;
            const float* input_ptr = input->data + input_offset;
            const float* filter_ptr = filter->data + filter_offset;
            float* output_ptr = output->data + output_offset;

            im2col_cpu(input_ptr, in_channels_per_group,
                       input_height, input_width, filter_height, filter_width,
                       padding_h_begin, padding_w_begin, stride_h, stride_w, dilation_h, dilation_w, data_col);

            // filter_ptr: (out_channels_per_group, in_channels_per_group * filter_height * filter_width)
            // data_col: (in_channels_per_group * filter_height * filter_width, output_height * output_width)
            // output_ptr: (out_channels_per_group, output_height * output_width)
            matmul(filter_ptr, data_col, output_ptr, 
                   out_channels_per_group, 
                   in_channels_per_group * filter_height * filter_width, 
                   output_height * output_width);
        }
    }
    delete [] data_col;
}

