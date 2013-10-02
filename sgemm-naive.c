#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>


#define blocksize 15

void sgemm( int m, int n, int d, float *A, float *C )
{
	#pragma omp parallel for
	
	for (int a = 0; a <= n; a+=blocksize)
		for (int b = 0; b <= m; b+=blocksize)
			for (int c = 0; c <= n; c+=blocksize)
				for( int i = a; i < n && i < a + blocksize; i++ )
					for( int k = b; k < m && k < b + blocksize; k++ )
						for( int j = c; j < n && j < c + blocksize; j++ ) 
				 			C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];

				
}
