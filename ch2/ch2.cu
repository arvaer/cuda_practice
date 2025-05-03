#include <cstdio>
#include <cstdlib>
#include <math.h>
__global__ void vecAddKernel ( float* A, float* B, float* C, int n ) {
	size_t i = threadIdx.x + blockDim.x * blockIdx.x;
	if ( i < n ) {
		C[i] = A[i] + B[i];
	}
}

void vecadd ( float* h_A, float* h_B, float* h_C, int n ) {
	int	   size = n * sizeof ( float );
	float *d_A, *d_B, *d_C;

	cudaMalloc ( (void**) &d_A, size );
	cudaMalloc ( (void**) &d_B, size );
	cudaMalloc ( (void**) &d_C, size );

	cudaMemcpy ( d_A, h_A, size, cudaMemcpyHostToDevice );
	cudaMemcpy ( d_B, h_B, size, cudaMemcpyHostToDevice );

	vecAddKernel<<<ceil ( n / 256.0 ), 256>>> ( d_A, d_B, d_C, n );

	cudaMemcpy ( h_C, d_C, size, cudaMemcpyDeviceToHost );
	cudaFree ( &d_A );
	cudaFree ( &d_B );
	cudaFree ( &d_C );
}

int main () {
	float* a = (float*) malloc ( 256 * sizeof ( float ) );
	float* b = (float*) malloc ( 256 * sizeof ( float ) );
	float* c = (float*) malloc ( 256 * sizeof ( float ) );

	for ( size_t i = 0; i < 256; ++i ) {
		a[i] = rand ();
		b[i] = rand ();
		c[i] = rand ();
	}

	vecadd ( a, b, c, 256 );

	printf ( "Printing C:\n" );
	for ( size_t i = 0; i < 256; ++i ) {
		printf ( "%f|", c[i] );
	}

	return 0;
}
