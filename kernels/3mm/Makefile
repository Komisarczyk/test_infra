TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=3mm-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: 3mm-mlir 

3mm-mlir: 3mm-mlir.s 3mm.s
	gcc -g -o 3mm-mlir.out main.s 3mm.s

3mm.s: 3mm.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm 3mm.mlir | mlir-translate -mlir-to-llvmir | llc > 3mm.s

3mm-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: 3mm-mlir
	$(BUILDDIR)/3mm-mlir.out
	
