TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=syrk-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: syrk-mlir 

syrk-mlir: syrk-mlir.s syrk.s
	gcc -g -o syrk-mlir.out main.s syrk.s

syrk.s: syrk.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm syrk.mlir | mlir-translate -mlir-to-llvmir | llc > syrk.s

syrk-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: syrk-mlir
	$(BUILDDIR)/syrk-mlir.out
	
