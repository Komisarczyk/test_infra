TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=gemver-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = gcc
BUILDDIR ?= .
all: gemver-mlir 

gemver-mlir: gemver-mlir.s gemver.s
	gcc -g -o gemver-mlir.out main.s gemver.s

gemver.s: gemver.mlir
	mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm gemver.mlir | mlir-translate -mlir-to-llvmir | llc > gemver.s

gemver-mlir.s: main.c
	gcc -g -S main.c

clean:
	rm -f *.s *.out

run: gemver-mlir
	$(BUILDDIR)/gemver-mlir.out
	
