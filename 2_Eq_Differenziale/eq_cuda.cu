#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MAXITER 1000000
#define TOLERANCE 0.000000001
#define NORM (N*N)


__global__ void diff_eq (float *A, int it, float *diff, int N) {
	int j;
	int i = (blockIdx.y * blockDim.y + threadIdx.y);
	float old;

	if((it%2) != (i%2)){
   		j = 2*(blockIdx.x * blockDim.x + threadIdx.x);		
   	}else{
   		j = 2*(blockIdx.x * blockDim.x + threadIdx.x) + 1;
   	}

	if((j >= N) || (i >= N)){
		return;
	}
	
	old = *(A + (N+2)*(i+1) + (j+1));

	*(A + (N+2)*(i+1) + (j+1)) =  0.2*(old) 
								+ 0.2*(*(A + (N+2)*(i+2) + (j+1)))
   								+ 0.2*(*(A + (N+2)*(i) + (j+1))) 
								+ 0.2*(*(A + (N+2)*(i+1) + (j+2))) 
								+ 0.2*(*(A + (N+2)*(i+1) + (j)));
	
	atomicAdd(diff, fabs(*(A + (N+2)*(i+1) + (j+1)) - old));

	return;
}

int main(void){
	float *A, *diff, gpuTime;
	int i, j, N, it;
	cudaEvent_t start, end;

	printf ("Dimensione della matrice escluse le condizioni al contorno (la matrice completa sarà qundi (N+2)*(N+2)): ");
  	scanf("%d", &N);

	cudaMallocManaged (&A, sizeof(float) * (N+2) * (N+2));
	cudaMallocManaged (&diff, sizeof(float));
	cudaDeviceSynchronize ();


	for(i = 1; i < (N+1); i++){
    	for(j = 1; j < (N+1); j++){
       		*(A + (N+2)*i + j) = 0;
   		}
 	}

 	for(i = 0; i < (N+2); i++){
    	*(A + (N+2)*i + 0) = i + 1;
   		*(A + (N+2)*i + (N+1)) = i + 1;
 	}

  	for(j = 0; j < (N+2); j++){
      	*(A + (N+2)*0 + j) = 1;
      	*(A + (N+2)*(N+1) + j) = N + 2;
  	}

	dim3 blocksPerGrid   ((int)(N/32 + 1), (int)(N/32 + 1), 1);
	dim3 threadsPerBlock (16, 32, 1);
	cudaEventCreate (&start);
  	cudaEventCreate (&end);
 	cudaEventRecord (start);
	do{
		it++;
    	*diff = 0;
 		diff_eq <<< blocksPerGrid, threadsPerBlock>>> (A, it, diff, N);
		cudaDeviceSynchronize();
		it++;
		diff_eq <<< blocksPerGrid, threadsPerBlock>>> (A, it, diff, N);
		cudaDeviceSynchronize();
	}while (((*diff / NORM) > TOLERANCE) && ((it / 2) < MAXITER));
 	cudaEventRecord (end);
 	cudaDeviceSynchronize ();

  	cudaEventElapsedTime (&gpuTime, start, end);

  	if(N <= 19){
		for(i = 0; i < (N+2); i++){
			for(j = 0; j < (N+2); j++){
				printf("%6.3f ",*(A + (N+2)*i + j));
			}
			printf("\n");
		}
  	}

	printf("\nIterazioni = %d\n", (int)(it / 2));
	printf ("Tempo impiegato: %.2f ms\n", gpuTime);
	
	cudaFree(A);
    cudaFree(diff);

	return 0;
}
