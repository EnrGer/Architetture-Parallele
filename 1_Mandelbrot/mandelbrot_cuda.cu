#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "bmp.h"

#define ITERATIONS 256
#define DOTS_PER_UNIT 2500
#define THREADX 16
#define THREADY 16

__constant__ COLORTRIPLE black = {0, 0, 0};
__constant__ COLORTRIPLE c0 = {0, 0, 51};
__constant__ COLORTRIPLE c1 = {0, 0, 153};
__constant__ COLORTRIPLE c2 = {0, 51, 255};
__constant__ COLORTRIPLE c3 = {51, 153, 255};
__constant__ COLORTRIPLE c4 = {0, 204, 255};
__constant__ COLORTRIPLE c5 = {51, 255, 255};
__constant__ COLORTRIPLE c6 = {0, 204, 0};
__constant__ COLORTRIPLE c7 = {255, 255, 0};
__constant__ COLORTRIPLE c8 = {255, 0, 0};


__global__ void mandelb (COLORTRIPLE *pixelPtr) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  double re, im;

  re = -2.0 + i*(1.0/DOTS_PER_UNIT);
  im = -1.0 + j*(1.0/DOTS_PER_UNIT);

  if((re >= 1) || (im >= 1)){
    return;
  }

  if ( (re*re + im*im) > 4){
    *(pixelPtr + j*3*DOTS_PER_UNIT + i) = c0;
    return;
  }else{
    double zkr = re;
    double zki = im;
    double zkr_prev , zki_prev;

    for (int n = 1; n <= ITERATIONS; n++){
      zkr_prev = zkr;
      zki_prev = zki;
      zkr = re + zkr_prev*zkr_prev - zki_prev*zki_prev ;
      zki = im + 2*zkr_prev*zki_prev ;

      if ((zkr*zkr + zki*zki) > 4){
        if(n == 1){
          *(pixelPtr + j*3*DOTS_PER_UNIT + i) = c1;
        }else if(n == 2){
          *(pixelPtr + j*3*DOTS_PER_UNIT + i) = c2;
        }else if(n == 3){
          *(pixelPtr + j*3*DOTS_PER_UNIT + i) = c3;
        }else if((n >= 4) && (n < 8)){
          *(pixelPtr + j*3*DOTS_PER_UNIT + i) = c4;
        }else if((n >= 8) && (n < 16)){
          *(pixelPtr + j*3*DOTS_PER_UNIT + i) = c5;
        }else if((n >= 16) && (n < 32)){
          *(pixelPtr + j*3*DOTS_PER_UNIT + i) = c6;
        }else if((n >= 32) && (n < 64)){
          *(pixelPtr + j*3*DOTS_PER_UNIT + i) = c7;
        }else if(n >= 64){
          *(pixelPtr + j*3*DOTS_PER_UNIT + i) = c8;
        }
        return;
      }
    }
    *(pixelPtr + j*3*DOTS_PER_UNIT + i) = black;
    return;
  }

}

void  WriteBitmap (BITMAP bitmap, FILE *fp)
{
   COLORTRIPLE triple;
   unsigned char fillbyte = 0;
   int nstep;
   int i, j, k, fill;

   fwrite (&bitmap.fileheader, sizeof (FILEHEADER), 1, fp);
   fwrite (&bitmap.bmpheader, sizeof (BMPHEADER), 1, fp);

   /* number of bytes in a row must be multiple of 4 */
   fill = bitmap.width % 4;

   for (i = 0; i < bitmap.height; i++)
   {
      for (j = 0; j < bitmap.width; j++)
      {
         nstep = j + (i * bitmap.width);
         triple = bitmap.pixel [nstep];
         fwrite (&triple, sizeof(COLORTRIPLE), 1, fp);
      }
      for (k = 0; k < fill; k++)
         fwrite (&fillbyte, sizeof(unsigned char), 1, fp);

   }

#ifdef BMPSHOWALL
   printf ("%d pixels written\n", nstep + 1);
#endif

   return;
}


void ReleaseBitmapData (BITMAP *bitmap)
{
   cudaFree ((*bitmap).pixel);
   (*bitmap).bmpheader.ImageHeight = (*bitmap).height = 0;
   (*bitmap).bmpheader.ImageWidth = (*bitmap).width = 0;
   (*bitmap).pixel = NULL;

   return;
}


BITMAP  CreateEmptyBitmap (dword height, dword width)
{
   BITMAP bitmap;

#ifdef BMPSHOWALL
   printf ("Creating empty bitmap %d x %d pixels\n", height, width);
#endif

   /* bitmap header */
   bitmap.fileheader.ImageFileType = BMPFILETYPE;   /* magic number! */
   bitmap.fileheader.FileSize = 14 + 40 + height * width * 3;
   bitmap.fileheader.Reserved1 = 0;
   bitmap.fileheader.Reserved2 = 0;
   bitmap.fileheader.ImageDataOffset = 14 + 40;

   /* bmp header */
   bitmap.bmpheader.HeaderSize = 40;
   bitmap.bmpheader.ImageWidth = bitmap.width = width;
   bitmap.bmpheader.ImageHeight = bitmap.height = height;
   bitmap.bmpheader.NumberOfImagePlanes = 1;
   bitmap.bmpheader.BitsPerPixel = 24;  /* the only supported format */
   bitmap.bmpheader.CompressionMethod = 0;  /* compression is not supported */
   bitmap.bmpheader.SizeOfBitmap = 0;  /* conventional value for uncompressed
                                          images */
   bitmap.bmpheader.HorizonalResolution = 0;  /* currently unused */
   bitmap.bmpheader.VerticalResolution = 0;  /* currently unused */
   bitmap.bmpheader.NumberOfColorsUsed = 0;  /* dummy value */
   bitmap.bmpheader.NumberOfSignificantColors = 0;  /* every color is
                                                       important */

   //bitmap.pixel = (COLORTRIPLE *) malloc (sizeof (COLORTRIPLE) * width * height);
   cudaMallocManaged (&(bitmap.pixel), sizeof (COLORTRIPLE) * width * height);
   if (bitmap.pixel == NULL)
   {
      printf ("Memory allocation error\n");
      exit (EXIT_FAILURE);
   }

   return bitmap;
}



int main(void){

  BITMAP bitmap;
  FILE *fpout;
  COLORTRIPLE *pixelPtr;
  cudaEvent_t start, end;
  float gpuTime;

  fpout = fopen("mandelbrot_cuda.bmp" , "wb");
  bitmap = CreateEmptyBitmap ((int)(2*DOTS_PER_UNIT), (int)(3*DOTS_PER_UNIT));
  pixelPtr = bitmap.pixel;

  cudaDeviceSynchronize();

  dim3 blocksPerGrid (1 + 3*DOTS_PER_UNIT/THREADX, 1 + 2*DOTS_PER_UNIT/THREADY);
  dim3 threadsPerBlock (THREADX, THREADY);

  cudaEventCreate (&start);
  cudaEventCreate (&end);
  cudaEventRecord (start);
  mandelb <<< blocksPerGrid, threadsPerBlock>>> (pixelPtr);
  cudaEventRecord (end);
  cudaDeviceSynchronize ();

  cudaEventElapsedTime (&gpuTime, start, end);
  printf ("Tempo impiegato: %.2f ms\n", gpuTime);

  WriteBitmap (bitmap, fpout);

  ReleaseBitmapData (&bitmap);
  fclose (fpout);

	return 0;
}
