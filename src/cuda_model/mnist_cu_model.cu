__global__ void convolution2d_kernel(
    const float* input,
    const float* weights,
    float* output,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size
) {
    // Compute output indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_height && col < output_width) {
        float sum = 0.0f;

        // Compute the convolution operation
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int input_row = row + i;
                int input_col = col + j;
                sum += input[input_row * input_width + input_col] *
                       weights[i * kernel_size + j];
            }
        }

        output[row * output_width + col] = sum;
    }
}