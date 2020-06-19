TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=mm-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: mm-mlir 

mm-mlir: mm-mlir.s gemm.s
	gcc -o mm-mlir.out main.s gemm.s

gemm.s: gemm.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm gemm.mlir | mlir-translate -mlir-to-llvmir | llc > gemm.s

mm-mlir.s: main.c
	gcc -S main.c

clean:
	rm -f *.s *.out

run: mm-mlir
	$(BUILDDIR)/mm-mlir.out
	
