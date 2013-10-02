#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>


#define i_dim 2
#define j_dim 2

void sgemm( int m, int n, int d, float *A, float *C )
{
	if (n == 40) {
		__m128 c[5][2], c_vector[5][2], trans, temp;
		float *padA;
		float *padC;
		
		int rem, old_n;

		if (n%(4*5*2) == 0) {
			rem = 0;
			padA = A;
			padC = C;
		} 
		else {
			rem = 4*5*2 - n%(4*5*2);
			int size = n*(d+n);
			int new_size = size + rem*(d+n);
			padA = (float*) malloc( new_size * sizeof(float) );
			padC = (float*) malloc( (n+rem) * (n+rem) * sizeof(float) );

			// float padA[new_size], padC[new_size];

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
		
		for( int i = 0; i < (n/(5*4))*5*4; i+=5*4 ) {

			for( int j = 0; j < (n/2)*2; j+=2 ) {

				for (int q = 0; q < 2; q++) {
					for (int p = 0; p < 5; p++) {
						c_vector[p][q] = _mm_loadu_ps((padC+(i+p*4)+(j+q)*n));
					}
				}

				for( int k = 0; k < m; k++ ) {

					for (int q = 0; q < 2; q++) {
						trans = _mm_load1_ps ((padA+((j+q)*(n+1)+k*n)));
						for (int p = 0; p < 5; p++) {
							temp = _mm_loadu_ps ((padA+(i+4*p)+k*n));
							c_vector[p][q] = _mm_add_ps(_mm_mul_ps(temp, trans), c_vector[p][q]);
						}
					}
			    }

			    for (int q = 0; q < 2; q++) {
					for (int p = 0; p < 5; p++) {
						_mm_storeu_ps ((padC+(i+p*4)+(j+q)*n), c_vector[p][q]);
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

	else {
		__m128 c[i_dim][j_dim], c_vector[i_dim][j_dim], trans, temp;
		float *padA;
		float *padC;
		
		int rem, old_n;

		if (n%(4*i_dim*j_dim) == 0) {
			rem = 0;
			padA = A;
			padC = C;
		} 
		else {
			rem = 4*i_dim*j_dim - n%(4*i_dim*j_dim);
			int size = n*(d+n);
			int new_size = size + rem*(d+n);
			padA = (float*) malloc( new_size * sizeof(float) );
			padC = (float*) malloc( (n+rem) * (n+rem) * sizeof(float) );

			// float padA[new_size], padC[new_size];

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
		
		for( int i = 0; i < (n/(i_dim*4))*i_dim*4; i+=i_dim*4 ) {

			for( int j = 0; j < (n/j_dim)*j_dim; j+=j_dim ) {

				for (int q = 0; q < j_dim; q++) {
					for (int p = 0; p < i_dim; p++) {
						c_vector[p][q] = _mm_loadu_ps((padC+(i+p*4)+(j+q)*n));
					}
				}

				for( int k = 0; k < m; k++ ) {

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
		
		if (rem != 0) {
			for (int col = 0; col < old_n; col++) {
				memcpy(C + old_n*col, padC+n*col, old_n*sizeof(float));
			}
			
			free(padA);
			free(padC);
		}
	}
}