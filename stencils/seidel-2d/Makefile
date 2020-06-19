TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=seidel-2d-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: seidel-2d-mlir 

seidel-2d-mlir: seidel-2d-mlir.s seidel-2d.s
	gcc -g -o seidel-2d-mlir.out main.s seidel-2d.s

seidel-2d.s: seidel-2d.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm seidel-2d.mlir | mlir-translate -mlir-to-llvmir | llc > seidel-2d.s

seidel-2d-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: seidel-2d-mlir
	$(BUILDDIR)/seidel-2d-mlir.out
	
