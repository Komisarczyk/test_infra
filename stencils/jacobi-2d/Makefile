TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=jacobi-2d-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: jacobi-2d-mlir 

jacobi-2d-mlir: jacobi-2d-mlir.s jacobi-2d.s
	gcc -g -o jacobi-2d-mlir.out main.s jacobi-2d.s

jacobi-2d.s: jacobi-2d.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm jacobi-2d.mlir | mlir-translate -mlir-to-llvmir | llc > jacobi-2d.s

jacobi-2d-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: jacobi-2d-mlir
	$(BUILDDIR)/jacobi-2d-mlir.out
	
