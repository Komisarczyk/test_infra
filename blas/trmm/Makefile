TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=trmm-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: trmm-mlir 

trmm-mlir: trmm-mlir.s trmm.s
	gcc -g -o trmm-mlir.out main.s trmm.s

trmm.s: trmm.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm trmm.mlir | mlir-translate -mlir-to-llvmir | llc > trmm.s

trmm-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: trmm-mlir
	$(BUILDDIR)/trmm-mlir.out
	
