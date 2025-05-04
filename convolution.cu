#include <cstdio>
#include <cstdlib>
#include <math.h>

#define TILE_WIDTH 16



__global__ void convolve(size_t m, size_t n, size_t f, float* X, float* F, float* A) {

}



void init_matrix ( size_t n, size_t m, float* A, float max ) {
	for ( size_t i = 0; i < n; ++i ) {
		for ( size_t j = 0; j < m; ++j ) {
			A[i * m + j] = (float) rand () / ( (float) RAND_MAX / max );
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
	size_t X_rows  = 150;
	size_t X_cols  = 150;

    size_t F_rows = 5;
    size_t F_cols = 5;

	float* h_X = (float*) malloc ( X_rows * X_cols * sizeof ( float ) );
	float* h_F = (float*) malloc ( F_rows * F_cols * sizeof ( float ) );
	float* h_A = (float*) malloc ( X_rows * X_cols * sizeof ( float ) );
	float *d_X, *d_F, *d_A;

	init_matrix ( X_rows, X_cols, h_X, 100.0 );
	init_matrix ( F_rows, F_cols, h_F, 100.0 );

	cudaMalloc ( (void**) &d_X, X_rows * X_cols * sizeof ( float ) );
	cudaMalloc ( (void**) &d_F, F_rows * F_cols * sizeof ( float ) );
	cudaMalloc ( (void**) &d_A, X_rows * X_cols * sizeof ( float ) );

	cudaMemcpy ( d_X, h_X, X_rows * X_cols * sizeof ( float ), cudaMemcpyHostToDevice );
	cudaMemcpy ( d_F, h_F, F_rows * F_cols * sizeof ( float ), cudaMemcpyHostToDevice );
	cudaMemset ( d_A, 0, X_rows * X_cols * sizeof ( float ) );

	int gx = ( X_rows + TILE_WIDTH - 1 ) / TILE_WIDTH;
	int gy = ( X_cols + TILE_WIDTH - 1 ) / TILE_WIDTH;
	dim3 blockDim ( TILE_WIDTH, TILE_WIDTH );
	dim3 gridDim ( gx, gy );

	convolve<<<gridDim, blockDim>>> ( X_rows, X_cols, F_rows, d_X, d_F, d_A );

    cudaDeviceSynchronize();
	cudaMemcpy ( h_A, d_A, X_cols * X_rows * sizeof ( float ), cudaMemcpyDeviceToHost );

	float base_result = calculate_result ( h_A, X_rows * X_cols );
	printf ( "Base Result: %f \n", base_result );



	return 0;
}
