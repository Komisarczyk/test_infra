TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=cholesky-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: cholesky-mlir 

cholesky-mlir: cholesky-mlir.s cholesky.s
	gcc -g -o cholesky-mlir.out main.s cholesky.s

cholesky.s: cholesky.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm cholesky.mlir | mlir-translate -mlir-to-llvmir | llc > cholesky.s

cholesky-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: cholesky-mlir
	$(BUILDDIR)/cholesky-mlir.out
	
