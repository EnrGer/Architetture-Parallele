#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define X 153
#define Y 42
#define DELAY 90
#define ROW 10
#define COL 64

__global__ void ga (char *A, char *B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int neighbours = 0;

    if( ((i+1) >= X) || ((i-1) < 0) || ((j+1) >= Y) || ((j-1) < 0) ){
        *(B + i + X*j) = '-';
    }else{ 
        if(*(A + (i+1) + X*j) == '*'){
            neighbours++;
        }
        if(*(A + (i-1) + X*j) == '*'){
            neighbours++;
        }
        if(*(A + i + X*(j+1)) == '*'){
            neighbours++;
        }
        if(*(A + i + X*(j-1)) == '*'){
            neighbours++;
        }
        if(*(A + (i+1) + X*(j+1)) == '*'){
            neighbours++;
        }
        if(*(A + (i+1) + X*(j-1)) == '*'){
            neighbours++;
        }
        if(*(A + (i-1) + X*(j+1)) == '*'){
            neighbours++;
        }
        if(*(A + (i-1) + X*(j-1)) == '*'){
            neighbours++;
        }
        if( *(A + i + X*j) == '*'){
            if((neighbours == 2) || (neighbours == 3)){
                *(B + i + X*j) = '*';
            }else{
                *(B + i + X*j) = '-';
            }
        }else{
            if((neighbours == 3)){
                *(B + i + X*j) = '*';
            }else{
                *(B + i + X*j) = '-';
            }
        }
    }

    return;
}

int main(void){

    char *A, *B, *C;
    int i, j, x;

    cudaMallocManaged (&A, X * Y * sizeof(char));
    cudaMallocManaged (&B, X * Y * sizeof(char));
	cudaDeviceSynchronize ();

    dim3 blocksPerGrid   (X, Y, 1);
	dim3 threadsPerBlock (1, 1, 1);

    for(j = 0; j < Y; j++){
        for(i = 0; i < X; i++){
                *(A + i + X*j) = '-';
        }
    }

    printf("select a configuration\n1. exploder\n2. 10 cell row\n3. gosper glider gun\n");
    x = getchar();
    if(x == '1'){
        for(j=20;j<25;j++){           //exploder
        *(A + 40 + X*j) = '*';
        *(A + 44 + X*j) = '*';
        }
        *(A + 42 + X*20) = '*';
        *(A + 42 + X*24) = '*';
    }else if (x == '2'){
        for(j=15,i=35;i<45;i++){         //10 cell row
        *(A + i + X*j) = '*';
    }
    }else if (x == '3'){
        *(A + (COL) + X*(ROW)) = '*';     //gosper glider gun
        *(A + (COL) + X*(ROW+1)) = '*';
        *(A + (COL-2) + X*(ROW+1)) = '*';
        *(A + (COL) + X*(ROW+5)) = '*';
        *(A + (COL) + X*(ROW+6)) = '*';
        *(A + (COL-3) + X*(ROW+2)) = '*';
        *(A + (COL-4) + X*(ROW+2)) = '*';
        *(A + (COL-4) + X*(ROW+3)) = '*';
        *(A + (COL-3) + X*(ROW+3)) = '*';
        *(A + (COL-3) + X*(ROW+4)) = '*';
        *(A + (COL-4) + X*(ROW+4)) = '*';
        *(A + (COL-2) + X*(ROW+5)) = '*';
        *(A + (COL+10) + X*(ROW+2)) = '*';
        *(A + (COL+10) + X*(ROW+3)) = '*';
        *(A + (COL+11) + X*(ROW+2)) = '*';
        *(A + (COL+11) + X*(ROW+3)) = '*';
        *(A + (COL-7) + X*(ROW+5)) = '*';
        *(A + (COL-8) + X*(ROW+4)) = '*';
        *(A + (COL-8) + X*(ROW+5)) = '*';
        *(A + (COL-8) + X*(ROW+6)) = '*';
        *(A + (COL-9) + X*(ROW+3)) = '*';
        *(A + (COL-9) + X*(ROW+7)) = '*';
        *(A + (COL-10) + X*(ROW+5)) = '*';
        *(A + (COL-11) + X*(ROW+2)) = '*';
        *(A + (COL-11) + X*(ROW+8)) = '*';
        *(A + (COL-12) + X*(ROW+2)) = '*';
        *(A + (COL-12) + X*(ROW+8)) = '*';
        *(A + (COL-13) + X*(ROW+3)) = '*';
        *(A + (COL-13) + X*(ROW+7)) = '*';
        *(A + (COL-14) + X*(ROW+4)) = '*';
        *(A + (COL-14) + X*(ROW+5)) = '*';
        *(A + (COL-14) + X*(ROW+6)) = '*';
        *(A + (COL-23) + X*(ROW+4)) = '*';
        *(A + (COL-23) + X*(ROW+5)) = '*';
        *(A + (COL-24) + X*(ROW+4)) = '*';
        *(A + (COL-24) + X*(ROW+5)) = '*';
    }else{
        printf("invalid selection\n");
        return 0;
    }
 
    while(1){
        system("clear");
        printf("\n");
        for(j = 3 ; j < (Y-3) ; j++){
            for(i = 3 ; i < (X-3); i++){
                printf("%c", *(A + i + X*j));
            }
            printf("\n");
        }
        ga <<< blocksPerGrid, threadsPerBlock>>> (A, B);
        cudaDeviceSynchronize ();
        C = A;
        A = B;
        B = C;
        usleep(DELAY*1000);
    }

    return 0;
}
