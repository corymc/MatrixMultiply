#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>


#define i_dim 5
#define j_dim 4
#define blocksize 600
#define MIN(X, Y) (X < Y ? X : Y)

void sgemm( int m, int n, int d, float *A, float *C )
{

	__m128 c_vector[i_dim][j_dim], trans, temp;
	float *padA;
	float *padC;
	
	int rem, old_n, new_size;

	if (n%(4*i_dim*j_dim) == 0) {
		rem = 0;
		padA = A;
		padC = C;
	} 
	else {
		rem = 4*i_dim*j_dim - n%(4*i_dim*j_dim);
		padA = (float*) malloc( (n+rem) * (n+d) * 2 * sizeof(float) );
		padC = (float*) malloc( (n+rem) * (n+rem) * sizeof(float) );

		for (int col = 0; col < d+n; col++) {
			memcpy( padA + (n+rem)*col, A+n*col, n*sizeof(float));
			memset( padA + (n+rem)*col + n, 0, rem*sizeof(float));
		}
		for (int col = 0; col < n; col++) {
			memcpy( padC + ((n+rem)*col), C+n*col, n*sizeof(float));
			memset( padC + (n+rem)*col + n, 0, rem*sizeof(float));
		}

		old_n = n;
		n = n+rem;

	}

	#pragma omp parallel private(c_vector, trans, temp)
	
	for (int a = 0; a < n; a+=blocksize) {

		for (int b = 0; b < n; b+=blocksize) {
			for (int c = 0; c < m; c+=blocksize) {
				#pragma omp for
				for( int j = a; j < MIN(n, a + blocksize); j+=j_dim ) {

					for( int i = b; i < MIN(n, b + blocksize); i+=i_dim*4 ) {
						for (int q = 0; q < j_dim; q++) {
							for (int p = 0; p < i_dim; p++) {
								c_vector[p][q] = _mm_loadu_ps((padC+(i+p*4)+(j+q)*n));
							}
						}

						for( int k = c; k < MIN(m, c + blocksize); k++ ) {

							for (int q = 0; q < j_dim; q++) {
								trans = _mm_load1_ps ((padA+((j+q)*(n+1)+k*n)));
								for (int p = 0; p < i_dim; p++) {
									temp = _mm_loadu_ps ((padA+(i+4*p)+k*n));
									c_vector[p][q] = _mm_add_ps(_mm_mul_ps(temp, trans), c_vector[p][q]);
								}
							}
					    }
					    for (int q = 0; q < j_dim; q++) {
							for (int p = 0; p < i_dim; p++) {
								_mm_storeu_ps ((padC+(i+p*4)+(j+q)*n), c_vector[p][q]);
							}
						}
					}
				}
			}
		}
	}

	if (rem != 0) {
		for (int col = 0; col < old_n; col++) {
			memcpy(C + old_n*col, padC+n*col, old_n*sizeof(float));
		}
		
		free(padA);
		free(padC);
	}
}