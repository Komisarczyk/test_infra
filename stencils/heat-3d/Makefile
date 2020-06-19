TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=heat-3d-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: heat-3d-mlir 

heat-3d-mlir: heat-3d-mlir.s heat-3d.s
	gcc -g -o heat-3d-mlir.out main.s heat-3d.s

heat-3d.s: heat-3d.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm heat-3d.mlir | mlir-translate -mlir-to-llvmir | llc > heat-3d.s

heat-3d-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: heat-3d-mlir
	$(BUILDDIR)/heat-3d-mlir.out
	
