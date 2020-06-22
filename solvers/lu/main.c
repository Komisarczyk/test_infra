#define ALPHA 1.0f
#define BETA 2.0f
#include "../../../polybench/linear-algebra/solvers/lu/lu.h"
#include "../../memref.h"
#include <stdio.h>
#include <string.h>

/* how to run me:

- mlir-opt -convert-linalg-to-affine-loops -lower-affine -convert-scf-to-std
  -convert-std-to-llvm gemm.mlir | mlir-translate -mlir-to-llvmir | llc > gemm.s

- gcc -S main.c

- gcc main.s gemm.s

- ./a.out


*/

/* Generated matrix multiplication function under test */
// See:
// https://mlir.llvm.org/docs/ConversionToLLVMDialect/#calling-convention-for-memref
extern void scop_entry(float *a_allocatedptr, float *a_alignedptr,
                       int64_t a_offset, int64_t a_sizes0, int64_t a_sizes1,
                       int64_t a_strides0, int64_t a_strides1);

/* Reference implementation of a matrix multiplication */
void mm_refimpl(struct vec_f2d *a) {

  for (int i = 0; i < a->sizes[0]; i++) {
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++) {
        vec_f2d_set(a, i, j, vec_f2d_get(a, i, j) -
                                 vec_f2d_get(a, i, k) * vec_f2d_get(a, k, j));
      }
      vec_f2d_set(a, i, j, vec_f2d_get(a, i, j) / vec_f2d_get(a, j, j));
    }
    for (int j = i; j < a->sizes[1]; j++) {
      for (int k = 0; k < i; k++) {
        vec_f2d_set(a, i, j, vec_f2d_get(a, i, j) -
                                 vec_f2d_get(a, i, k) * vec_f2d_get(a, k, j));
      }
    }
  }
}

/* Initialize vector with value x at position (x) */
void init_vector(struct vec_f1d *v) {
  for (int64_t x = 0; x < v->sizes[0]; x++)
    vec_f1d_set(v, x, x);
}

/* Initialize matrix with value x+y at position (x, y) */
void init_matrix(struct vec_f2d *m) {
  for (int64_t y = 0; y < m->sizes[1]; y++)
    for (int64_t x = 0; x < m->sizes[0]; x++)
      vec_f2d_set(m, x, y, x + y);
}

void die_usage(const char *program_name) {
  fprintf(stderr, "Usage: %s [-v]\n", program_name);
  exit(1);
}

int main(int argc, char **argv) {
  struct vec_f2d a, a_ref;
  int verbose = 0;
  int n = N;

  if (vec_f2d_alloc(&a, n, n) || vec_f2d_alloc(&a_ref, n, n)) {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_matrix(&a_ref);

  if (verbose) {
    puts("A:");
    vec_f2d_dump(&a);
    puts("");

    puts("A:");
    vec_f2d_dump(&a_ref);
    puts("");
  }

  scop_entry(VEC2D_ARGS(&a));

  mm_refimpl(&a_ref);

  if (verbose) {
    puts("Result O:");
    vec_f2d_dump(&a);
    puts("");

    puts("Reference O:");
    vec_f2d_dump(&a_ref);
    puts("");
  }

  if (!vec_f2d_compare(&a, &a_ref)) {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&a);
  vec_f2d_destroy(&a_ref);
  return 0;
}
