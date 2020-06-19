TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=gesummv-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: gesummv-mlir 

gesummv-mlir: gesummv-mlir.s gesummv.s
	gcc -g -o gesummv-mlir.out main.s gesummv.s

gesummv.s: gesummv.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm gesummv.mlir | mlir-translate -mlir-to-llvmir | llc > gesummv.s

gesummv-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: gesummv-mlir
	$(BUILDDIR)/gesummv-mlir.out
	
