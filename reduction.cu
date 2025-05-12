#include <cstdio>
__global__ void sum_reduction_simple(float* input, float* output) {
    extern __shared__ float sdata[];
    unsigned tid = threadIdx.x;
    sdata[tid] = input[tid];
    __syncthreads();

    // reduction in shared memory
    for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
        unsigned index = 2 * stride * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = sdata[0];
    }
}

__global__ void sum_reduction_simple_2 ( float* input, float* output ) {
	int tx = threadIdx.x;
	for ( size_t stride = blockDim.x; stride >= 1  ; stride /= 2 ) {
		if ( threadIdx.x % stride == 0 ) {
			input[tx] += input[tx+stride];
		}
		__syncthreads();
	}

	if ( threadIdx.x == 0 ) {
		*output = input[0];
	}
}

__global__ void segmented_reduction( float* input, float* output) {
    extern __shared__ float input_s[];
    // so we assign segments based off this. and then each block gets a segmnet of data that it processes. At the end, we atomic add from input and store output at [0]
   int segment = blockIdx.x * blockDim.x * 2;
   int tix = segment + threadIdx.x;
   int i = threadIdx.x;
    // first thing we do is load into input_s from input
    //
    // basically input_s is shared memory of a segment, ffrom which we are pulling tix + blockDim.x in
    input_s[i] = input[tix] + input[tix + blockDim.x];
    for(int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[i] += input_s[i + stride];
        }
    }

    if(tix == 0) {
        atomicAdd(output, input_s[0]);
    }
}

void init_vector(float* A, size_t n, float max) {
    for (size_t i = 0; i < n; ++i)
        A[i] = static_cast<float>(rand()) / (RAND_MAX / max);
}

int main() {
    const size_t len = 32;             // must be power-of-two for this reduction
    float *h_in  = (float*)malloc(len * sizeof(float));
    float  h_out;
    float *d_in, *d_out;

    init_vector(h_in, len, 100.0f);
    cudaMalloc(&d_in,  len * sizeof(float));
    cudaMalloc(&d_out,       sizeof(float));

    cudaMemcpy(d_in, h_in, len * sizeof(float), cudaMemcpyHostToDevice);

    // one block of len threads; allocate len*sizeof(float) shared memory
    sum_reduction_simple<<<1, len, len * sizeof(float)>>>(d_in, d_out);

    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum is %f\n", h_out);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}
