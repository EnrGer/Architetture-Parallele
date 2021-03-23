#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define TOLERANCE 0.000000001
#define MAXITER 1000000
#define NORM (N*N)

int N;

int solve (float *A) {
  float diff, old;
  int it = 0;
  int i, j;

  do{
    it++;
    diff = 0;

    for (i = 1; i < N + 1; i++) {
      for (j = 1; j < N + 1; j++) {
        old = *(A + (N+2)*i + j);
        *(A + (N+2)*i + j) = 0.2 * (*(A + (N+2)*(i+1) + j) 
                                  + *(A + (N+2)*(i-1) + j) 
                                  + old 
                                  + *(A + (N+2)*i + (j+1)) 
                                  + *(A + (N+2)*i + (j-1)));
        diff += fabs (*(A + (N+2)*i + j) - old);
      }
    }
  }while (((diff / NORM) > TOLERANCE) && (it < MAXITER));

  return it;
}

int main(void){
  float *A;
  int i, j, n;
  double time;
  struct timespec start, end;
  printf ("Dimensione della matrice escluse le condizioni al contorno (la matrice completa sarà qundi (N+2)*(N+2)): ");
  scanf("%d", &N);
  A = (float *) malloc ((N+2)*(N+2)*sizeof(float));

  for(i = 1; i < (N+1); i++){
    for(j = 1; j < (N+1); j++){
        *(A + (N+2)*i + j) = 0;
    }
  }

  for(i = 0; i < (N+2); i++){
    *(A + (N+2)*i + 0) = i+1;
    *(A + (N+2)*i + (N+1)) = i+1;
  }

  for(j = 0; j < (N+2); j++){
      *(A + (N+2)*0 + j) = 1;
      *(A + (N+2)*(N+1) + j) = N+2;
  }

  clock_gettime (CLOCK_REALTIME, &start);
  n = solve(A);
  clock_gettime (CLOCK_REALTIME, &end);
  time = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;
  
  if(N <= 19){
    for(i = 0; i < (N+2); i++){
      for(j = 0; j < (N+2); j++){
        printf("%6.3f ",*(A + (N+2)*i + j));
      }
      printf("\n");
    }
  }

  printf("\nIterazioni = %d\n", n);
  printf("Tempo impiegato: %.2f ms\n", time);

  free(A);

	return 0;
}
