TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=jacobi-1d-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: jacobi-1d-mlir 

jacobi-1d-mlir: jacobi-1d-mlir.s jacobi-1d.s
	gcc -g -o jacobi-1d-mlir.out main.s jacobi-1d.s

jacobi-1d.s: jacobi-1d.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm jacobi-1d.mlir | mlir-translate -mlir-to-llvmir | llc > jacobi-1d.s

jacobi-1d-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: jacobi-1d-mlir
	$(BUILDDIR)/jacobi-1d-mlir.out
	
