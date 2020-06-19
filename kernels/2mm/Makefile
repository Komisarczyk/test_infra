TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=2mm-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: 2mm-mlir 

2mm-mlir: 2mm-mlir.s 2mm.s
	gcc -g -o 2mm-mlir.out main.s 2mm.s

2mm.s: 2mm.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm 2mm.mlir | mlir-translate -mlir-to-llvmir | llc > 2mm.s

2mm-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: 2mm-mlir
	$(BUILDDIR)/2mm-mlir.out
	
