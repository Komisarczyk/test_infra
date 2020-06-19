TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=lu-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: lu-mlir 

lu-mlir: lu-mlir.s lu.s
	gcc -g -o lu-mlir.out main.s lu.s

lu.s: lu.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm lu.mlir | mlir-translate -mlir-to-llvmir | llc > lu.s

lu-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: lu-mlir
	$(BUILDDIR)/lu-mlir.out
	
