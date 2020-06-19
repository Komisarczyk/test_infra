TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=trisolv-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: trisolv-mlir 

trisolv-mlir: trisolv-mlir.s trisolv.s
	gcc -g -o trisolv-mlir.out main.s trisolv.s

trisolv.s: trisolv.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm trisolv.mlir | mlir-translate -mlir-to-llvmir | llc > trisolv.s

trisolv-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: trisolv-mlir
	$(BUILDDIR)/trisolv-mlir.out
	
