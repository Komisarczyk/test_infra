TFLAGS=-O3
MFLAGS= -O3 -e scop_entry --entry-point-result=void -object-filename=mm-mlir.o
OPTFLAGS = -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm
CFLAGS=$(TFLAGS)
CC = clang
BUILDDIR ?= .
IDIR ?= /home/konrad/Documents/llvm-project/mlir/tools/mlir-pet/Test
LIBDIR ?= /home/konrad/Documents/llvm-project/build/lib/
all: mm-mlir 

linked: kernelTest.bc mm-mlir.bc
	llvm-link -o=linked kernelTest.bc mm-mlir.bc 
 
assembly.s: linked 
	llc -o=assembly.s linked

mm-mlir: assembly.s
	clang -o mm-mlir -g assembly.s

kernelTest.bc: kernelTest.c
	clang -O3 -emit-llvm -c -o kernelTest.bc kernelTest.c

mm-mlir.bc: mm-mlir1.s
	llvm-as -o=mm-mlir.bc mm-mlir1.s
mm-mlir1.s: mm-mlir.s
	sed '2i\ target datalayout= "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"' mm-mlir.s > mm-mlir1.s

mm-mlir.s: gemmB.mlir
	mlir-opt -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm gemmB.mlir | mlir-translate -o=mm-mlir.s --mlir-to-llvmir 

clean:
	rm -f *.o *.bc *.s mm-mlir linked

run:
	$(BUILDDIR)/mm-mlir
	
#mm-mlir.s: gemmB.mlir
#	mlir-opt -convert-linalg-to-loops -convert-linalg-to-llvm -convert-std-to-llvm /home/konrad/Documents/llvm-project/mlir/tools/mlir-pet/Test/gemmB.mlir | mlir-translate -o=mm-mlir.s --mlir-to-llvmir 
