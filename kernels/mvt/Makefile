TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=mvt-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: mvt-mlir 

mvt-mlir: mvt-mlir.s mvt.s
	gcc -g -o mvt-mlir.out main.s mvt.s

mvt.s: mvt.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm mvt.mlir | mlir-translate -mlir-to-llvmir | llc > mvt.s

mvt-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: mvt-mlir
	$(BUILDDIR)/mvt-mlir.out
	
