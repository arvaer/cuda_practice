#include <math.h>
#include <cstdio>
#include <cstdlib>

#define TILE_WIDTH 16

__global__ void matmul ( size_t m, size_t n, size_t k, float* d_M, float* d_N, float* d_S ) {
	int   row = blockDim.y * blockIdx.y + threadIdx.y;
	int   col = blockDim.x * blockIdx.x + threadIdx.x;
	float sum = 0.0f;

	if ( row < m ) {
		if ( col < n ) {
			for ( size_t a = 0; a < k; ++a ) {
				sum += d_M[row * k + a] * d_N[a * n + col];
			}
		}
	}

	d_S[row * n + col] = sum;
}

__global__ void matmul_co ( size_t m,
                            size_t n,
                            size_t k,
                            const float* __restrict__ d_M,
                            const float* __restrict__ d_N,
                            float* d_S ) {
	// initialize shared memory
	// determine row and column based on block and thread id and tile width
	__shared__ float s_M[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_N[TILE_WIDTH][TILE_WIDTH];

	int   row = TILE_WIDTH * blockIdx.y + threadIdx.y;
	int   col = TILE_WIDTH * blockIdx.x + threadIdx.x;
	float sum = 0.0f;

	for ( int t = 0; t < (int)k; t += TILE_WIDTH ) {
		if ( row < m && ( t + threadIdx.x ) < k ) {
			s_M[threadIdx.y][threadIdx.x] = d_M[row * k + ( t + threadIdx.x )];
		} else {
			s_M[threadIdx.y][threadIdx.x] = 0.0f;
		}

		if ( col < n && ( t + threadIdx.y ) < n ) {
			s_N[threadIdx.y][threadIdx.x] = d_N[( t + threadIdx.y ) * n + col];
		} else {
			s_N[threadIdx.y][threadIdx.x] = 0.0f;
		}

		__syncthreads();

#pragma unroll
		for ( size_t dragon = 0; dragon < TILE_WIDTH; ++dragon ) {
			sum += s_M[threadIdx.x][dragon] * s_N[threadIdx.y][col];
		}

		__syncthreads();
	}
	if ( row < m && col < n )
		d_S[row * n + col] = sum;
}

void init_matrix ( size_t n, size_t m, float* A, float max ) {
	for ( size_t i = 0; i < n; ++i ) {
		for ( size_t j = 0; j < m; ++j ) {
			A[i * m + j] = (float)rand() / ( (float)RAND_MAX / max );
		}
	}
}

float calculate_result ( float* array, size_t size ) {
	float result = 0.0f;
	for ( int i = 0; i < size; ++i ) {
		result += array[i];
	}

	return result;
}

int main () {
	// initialize three matricies
	size_t M_rows  = 150;
	size_t N_cols  = 2500;
	size_t K_width = 1072;

	float* h_M = (float*)malloc( M_rows * K_width * sizeof( float ) );
	float* h_N = (float*)malloc( K_width * N_cols * sizeof( float ) );
	float* h_S = (float*)malloc( M_rows * N_cols * sizeof( float ) );
	float *d_M, *d_N, *d_S;

	init_matrix( M_rows, K_width, h_M, 100.0 );
	init_matrix( K_width, N_cols, h_N, 100.0 );

	cudaMalloc( (void**)&d_M, M_rows * K_width * sizeof( float ) );
	cudaMalloc( (void**)&d_N, K_width * N_cols * sizeof( float ) );
	cudaMalloc( (void**)&d_S, M_rows * N_cols * sizeof( float ) );

	cudaMemcpy( d_M, h_M, M_rows * K_width * sizeof( float ), cudaMemcpyHostToDevice );
	cudaMemcpy( d_N, h_N, K_width * N_cols * sizeof( float ), cudaMemcpyHostToDevice );
	cudaMemset( d_S, 0, M_rows * N_cols * sizeof( float ) );

	int  gx = ( M_rows + TILE_WIDTH - 1 ) / TILE_WIDTH;
	int  gy = ( N_cols + TILE_WIDTH - 1 ) / TILE_WIDTH;
	dim3 blockDim( TILE_WIDTH, TILE_WIDTH );
	dim3 gridDim( gx, gy );

	matmul<<<gridDim, blockDim>>>( M_rows, N_cols, K_width, d_M, d_N, d_S );

	cudaDeviceSynchronize();
	cudaMemcpy( h_S, d_S, N_cols * M_rows * sizeof( float ), cudaMemcpyDeviceToHost );

	float base_result = calculate_result( h_S, M_rows * N_cols );
	printf( "Base Result: %f \n", base_result );

	cudaMemset( d_S, 0, M_rows * N_cols * sizeof( float ) );
	matmul_co<<<gridDim, blockDim>>>( M_rows, N_cols, K_width, d_M, d_N, d_S );

	cudaDeviceSynchronize();
	cudaMemcpy( h_S, d_S, N_cols * M_rows * sizeof( float ), cudaMemcpyDeviceToHost );

	base_result = calculate_result( h_S, M_rows * N_cols );
	printf( "Base Result: %f \n", base_result );


	return 0;
}
