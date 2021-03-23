#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include "bmp.h"

#define DOTS_PER_UNIT 2500
#define ITERATIONS 256

COLORTRIPLE black = {0, 0, 0};
COLORTRIPLE c0 = {0, 0, 51};
COLORTRIPLE c1 = {0, 0, 153};
COLORTRIPLE c2 = {0, 51, 255};
COLORTRIPLE c3 = {51, 153, 255};
COLORTRIPLE c4 = {0, 204, 255};
COLORTRIPLE c5 = {51, 255, 255};
COLORTRIPLE c6 = {0, 204, 0};
COLORTRIPLE c7 = {255, 255, 0};
COLORTRIPLE c8 = {255, 0, 0};
BITMAP bitmap;
int num_cores;
double span;

int mandelb (double r , double i){

  if ( (r*r + i*i) > 4){
    return 1;
  }

  double zkr = r;
  double zki = i;
  double zkr_prev , zki_prev;
  int n;

  for (n=1; (n<=ITERATIONS) ; n++){
    zkr_prev = zkr;
    zki_prev = zki;

    zkr = r + zkr_prev*zkr_prev - zki_prev*zki_prev ;
    zki = i + 2*zkr_prev*zki_prev ;

    if ((zkr*zkr + zki*zki) > 4){
        return (n+1);
    }
  }

  return 0;
}

void *my_thread_start (void *arg) {

  printf("Thread: %2d  -  Core:  %2d\n" , arg , sched_getcpu());
  int offset, n, r, i;
  offset = arg;
  
  for ( r = offset*3*DOTS_PER_UNIT/num_cores ; r < (offset+1)*3*DOTS_PER_UNIT/num_cores ; r++){
    for ( i = 0 ; i < 2*DOTS_PER_UNIT ; i++){  
      n = mandelb((double)r/DOTS_PER_UNIT - 2, (double)i/DOTS_PER_UNIT -1 );
      if(n > 0){
        if(n == 1){
          PIXEL(bitmap , i , r ) = c0;
        }else if(n == 2){
          PIXEL(bitmap , i , r ) = c1;
        }else if(n == 3){
          PIXEL(bitmap , i , r ) = c2;
        }else if(n == 4){
          PIXEL(bitmap , i , r ) = c3;
        }else if((n >= 5) && (n < 9)){
          PIXEL(bitmap , i , r ) = c4;
        }else if((n >= 9) && (n < 17)){
          PIXEL(bitmap , i , r ) = c5;
        }else if((n >= 17) && (n < 33)){
          PIXEL(bitmap , i , r ) = c6;
        }else if((n >= 33) && (n < 65)){
          PIXEL(bitmap , i , r ) = c7;
        }else if(n >= 65){
          PIXEL(bitmap , i , r ) = c8;
        }
      }else{
        PIXEL(bitmap , i , r ) = black;
      }
    }
  }

  pthread_exit(NULL);
}

int main(void){

  int j;
  double time;
  struct timespec start, end;
  FILE *fpout;

  num_cores = sysconf(_SC_NPROCESSORS_ONLN);
  pthread_t threads[num_cores];
  cpu_set_t cpu;
  pthread_attr_t attr;
  pthread_attr_init(&attr);

  span = (double)3/num_cores;

  fpout = fopen("mandelbrot_thread.bmp" , "wb");
  bitmap = CreateEmptyBitmap ((int)(2*DOTS_PER_UNIT), (int)(3*DOTS_PER_UNIT));

  clock_gettime (CLOCK_REALTIME, &start);
  for (j=0; j<num_cores; j++){
    CPU_ZERO(&cpu);
    CPU_SET(j, &cpu);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpu);
    pthread_create (&threads[j], &attr, &my_thread_start, j);
  }
  for (j=0; j<num_cores; j++){
    pthread_join (threads[j], NULL);
  }
  clock_gettime (CLOCK_REALTIME, &end);

  time = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;
  printf("Tempo impiegato: %.2f ms\n", time);

  WriteBitmap (bitmap, fpout);
  ReleaseBitmapData (&bitmap);
  fclose (fpout);

	return 0;
}