TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=atax-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: atax-mlir 

atax-mlir: atax-mlir.s atax.s
	gcc -g -o atax-mlir.out main.s atax.s

atax.s: atax.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm atax.mlir | mlir-translate -mlir-to-llvmir | llc > atax.s

atax-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: atax-mlir
	$(BUILDDIR)/atax-mlir.out
	
