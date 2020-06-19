TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=syr2k-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: syr2k-mlir 

syr2k-mlir: syr2k-mlir.s syr2k.s
	gcc  -o syr2k-mlir.out main.s syr2k.s

syr2k.s: syr2k.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm syr2k.mlir | mlir-translate -mlir-to-llvmir | llc > syr2k.s

syr2k-mlir.s: main.c
	gcc  -S main.c

clean:
	rm -f *.s *.out

run: syr2k-mlir
	$(BUILDDIR)/syr2k-mlir.out
	
