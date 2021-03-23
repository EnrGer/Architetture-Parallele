#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bmp.h"

#define DOTS_PER_UNIT 2500
#define ITERATIONS 256

int mandelb (double r , double i){

  if ( (r*r + i*i) > 4){
    return 1;
  }

  double zkr = r;
  double zki = i;
  double zkr_prev, zki_prev;
  int n;

  for (n = 1 ; (n <= ITERATIONS) ; n++){
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


int main(void){

  double r, i;
  double step = 1.0/DOTS_PER_UNIT;
  int n;
  struct timespec start, end;
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
  FILE *fpout;

  fpout = fopen("mandelbrot.bmp" , "wb");
  bitmap = CreateEmptyBitmap ((int)(2*DOTS_PER_UNIT), (int)(3*DOTS_PER_UNIT));

  clock_gettime (CLOCK_REALTIME, &start);

  for ( r = -2 ; r < 1 ; r = r + step){
    for ( i = -1 ; i < 1 ; i = i + step){ 
      n = mandelb(r, i);
      if(n > 0){
        if(n == 1){
          PIXEL(bitmap , (int)((i+1)*DOTS_PER_UNIT) , (int)((r+2)*DOTS_PER_UNIT) ) = c0;
        }else if(n == 2){
          PIXEL(bitmap , (int)((i+1)*DOTS_PER_UNIT) , (int)((r+2)*DOTS_PER_UNIT) ) = c1;
        }else if(n == 3){
          PIXEL(bitmap , (int)((i+1)*DOTS_PER_UNIT) , (int)((r+2)*DOTS_PER_UNIT) ) = c2;
        }else if(n == 4){
          PIXEL(bitmap , (int)((i+1)*DOTS_PER_UNIT) , (int)((r+2)*DOTS_PER_UNIT) ) = c3;
        }else if((n >= 5) && (n < 9)){
          PIXEL(bitmap , (int)((i+1)*DOTS_PER_UNIT) , (int)((r+2)*DOTS_PER_UNIT) ) = c4;
        }else if((n >= 9) && (n < 17)){
          PIXEL(bitmap , (int)((i+1)*DOTS_PER_UNIT) , (int)((r+2)*DOTS_PER_UNIT) ) = c5;
        }else if((n >= 17) && (n < 33)){
          PIXEL(bitmap , (int)((i+1)*DOTS_PER_UNIT) , (int)((r+2)*DOTS_PER_UNIT) ) = c6;
        }else if((n >= 33) && (n < 65)){
          PIXEL(bitmap , (int)((i+1)*DOTS_PER_UNIT) , (int)((r+2)*DOTS_PER_UNIT) ) = c7;
        }else if(n >= 65){
          PIXEL(bitmap , (int)((i+1)*DOTS_PER_UNIT) , (int)((r+2)*DOTS_PER_UNIT) ) = c8;
        }
      }else{
        PIXEL(bitmap , (int)((i+1)*DOTS_PER_UNIT) , (int)((r+2)*DOTS_PER_UNIT) ) = black;
      }
    }
  }

  clock_gettime (CLOCK_REALTIME, &end);

  double time = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;
  printf("Tempo impiegato: %.2f ms\n", time);

  WriteBitmap (bitmap , fpout);
  ReleaseBitmapData (&bitmap);
  fclose (fpout);

	return 0;
}