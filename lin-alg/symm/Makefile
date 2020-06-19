TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=symm-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: symm-mlir 

symm-mlir: symm-mlir.s symm.s
	gcc -g -o symm-mlir.out main.s symm.s

symm.s: symm.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm symm.mlir | mlir-translate -mlir-to-llvmir | llc > symm.s

symm-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: symm-mlir
	$(BUILDDIR)/symm-mlir.out
	
