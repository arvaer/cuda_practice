#include <math.h>
#include <cstdio>
#include <cstdlib>

#define TILE_WIDTH 16
#define FILT_RADIUS 3
__constant__ float Filter[2 * FILT_RADIUS + 1][2 * FILT_RADIUS + 1];

__global__ void convolve ( size_t m, size_t n, size_t f, float* __restrict__ X, float* __restrict__ F, float* __restrict__ A ) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	// we know that f is square. so basically we need to calclulate the midpoint to be able to scale
	float sum = 0.0f;
	if ( row >= m || col >= n )
		return;
	if ( f % 2 == 0 ) {
		// how do I raise an error here?
		return;
	}
	int f_mid = f / 2;
	for ( int dy = -f_mid; dy <= f_mid; ++dy ) {
		for ( int dx = -f_mid; dx <= f_mid; ++dx ) {
			int y = row + dy;
			int x = col + dx;

			if ( y >= 0 && y < m && x >= 0 && x < n ) {
				float X_val = X[y * n + x];
				float F_val = F[( f_mid + dy ) * f + ( f_mid + dx )];
				sum += X_val * F_val;
			}
		}
	}

	A[row * n + col] = sum;
}

__global__ void convolve_constant_mem ( size_t m, size_t n, float* __restrict__ X, float* __restrict__ A ) {
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	// we know that f is square. so basically we need to calclulate the midpoint to be able to scale
	float sum = 0.0f;
	if ( row >= m || col >= n )
		return;

	for ( int dy = -FILT_RADIUS; dy <= FILT_RADIUS; ++dy ) {
		for ( int dx = -FILT_RADIUS; dx <= FILT_RADIUS; ++dx ) {
			int y = row + dy;
			int x = col + dx;

			if ( y >= 0 && y < m && x >= 0 && x < n ) {
				float X_val = X[y * n + x];
				float F_val = Filter[FILT_RADIUS + dy][FILT_RADIUS + dx];
				sum += X_val * F_val;
			}
		}
	}

	A[row * n + col] = sum;
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
	size_t X_rows = 10000;
	size_t X_cols = 10000;

	size_t filter_size = 2 * FILT_RADIUS + 1;

	size_t gx = ( X_rows + TILE_WIDTH - 1 ) / TILE_WIDTH;
	size_t gy = ( X_cols + TILE_WIDTH - 1 ) / TILE_WIDTH;

	float* h_X = (float*)malloc( X_rows * X_cols * sizeof( float ) );
	float* h_F = (float*)malloc( filter_size * filter_size * sizeof( float ) );
	float* h_A = (float*)malloc( X_rows * X_cols * sizeof( float ) );
	float *d_X, *d_F, *d_A;

	dim3 blockDim( TILE_WIDTH, TILE_WIDTH );
	dim3 gridDim( gx, gy );

	init_matrix( X_rows, X_cols, h_X, 100.0 );
	init_matrix( filter_size, filter_size, h_F, 100.0 );


	cudaMalloc( (void**)&d_X, X_rows * X_cols * sizeof( float ) );
	cudaMalloc( (void**)&d_F, filter_size * filter_size * sizeof( float ) );
	cudaMalloc( (void**)&d_A, X_rows * X_cols * sizeof( float ) );

	cudaMemcpy( d_X, h_X, X_rows * X_cols * sizeof( float ), cudaMemcpyHostToDevice );
	cudaMemcpy( d_F, h_F, filter_size * filter_size * sizeof( float ), cudaMemcpyHostToDevice );

	cudaMemcpyToSymbol( Filter, h_F, filter_size * filter_size * sizeof( float ) );
	cudaMemset( d_A, 0, X_rows * X_cols * sizeof( float ) );

	cudaEvent_t start, stop;
	float       elapsed = 0.0f;

	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	cudaEventRecord( start );
	convolve<<<gridDim, blockDim>>>( X_rows, X_cols, filter_size, d_X, d_F, d_A );
	cudaError_t err = cudaGetLastError();
	if ( err != cudaSuccess ) {
		printf( "CUDA Error: %s\n", cudaGetErrorString( err ) );
	}

	cudaEventRecord( stop );

	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed, start, stop );

	cudaMemcpy( h_A, d_A, X_cols * X_rows * sizeof( float ), cudaMemcpyDeviceToHost );

	float base_result = calculate_result( h_A, X_rows * X_cols );
	printf( "Base Result (regular): %f\n", base_result );
	printf( "Kernel Time (regular): %.3f ms\n", elapsed );


	cudaMemset( d_A, 0, X_rows * X_cols * sizeof( float ) );

	cudaEventRecord( start );
	convolve_constant_mem<<<gridDim, blockDim>>>( X_rows, X_cols, d_X, d_A );
	err = cudaGetLastError();
	if ( err != cudaSuccess ) {
		printf( "CUDA Error: %s\n", cudaGetErrorString( err ) );
	}

	cudaEventRecord( stop );

	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed, start, stop );

	cudaMemcpy( h_A, d_A, X_cols * X_rows * sizeof( float ), cudaMemcpyDeviceToHost );

	base_result = calculate_result( h_A, X_rows * X_cols );
	printf( "Base Result (const mem): %f\n", base_result );
	printf( "Kernel Time (const mem): %.3f ms\n", elapsed );

	cudaEventDestroy( start );
	cudaEventDestroy( stop );


	cudaFree( d_X );
	cudaFree( d_F );
	cudaFree( d_A );
	free( h_X );
	free( h_F );
	free( h_A );

	return 0;
}
