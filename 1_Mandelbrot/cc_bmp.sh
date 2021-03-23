#!/bin/sh
gcc -o$1 -DBMPSHOWALL $1.c bmp.c -lm
