TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=bicg-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: bicg-mlir 

bicg-mlir: bicg-mlir.s bicg.s
	gcc -g -o bicg-mlir.out main.s bicg.s

bicg.s: bicg.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm bicg.mlir | mlir-translate -mlir-to-llvmir | llc > bicg.s

bicg-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: bicg-mlir
	$(BUILDDIR)/bicg-mlir.out
	
