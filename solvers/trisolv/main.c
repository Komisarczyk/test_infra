#define ALPHA 1.0f
#define BETA 2.0f
#include "../../../polybench/linear-algebra/solvers/trisolv/trisolv.h"
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
                       int64_t a_strides0, int64_t a_strides1,

                       float *p_allocatedptr, float *p_alignedptr,
                       int64_t p_offset, int64_t p_sizes0, int64_t p_strides0,

                       float *r_allocatedptr, float *r_alignedptr,
                       int64_t r_offset, int64_t r_sizes0, int64_t r_strides0);

/* Reference implementation of a matrix multiplication */
static void mm_refimpl(struct vec_f2d *a, struct vec_f1d *x,
                       struct vec_f1d *b) {

#pragma scop
  for (int i = 0; i < x->sizes[0]; i++) {
    vec_f1d_set(x, i, vec_f1d_get(b, i));
    for (int j = 0; j < i; j++)
      vec_f1d_set(x, i,
                  vec_f1d_get(x, i) - vec_f2d_get(a, i, j) * vec_f1d_get(x, j));
    vec_f1d_set(x, i, vec_f1d_get(x, i) / vec_f2d_get(a, i, i));
  }
#pragma endscop
}

/* Initialize vector with value x at position (x) */
void init_vector(struct vec_f1d *v) {
  for (int64_t x = 0; x < v->sizes[0]; x++)
    vec_f1d_set(v, x, 1 + x * x);
}

/* Initialize matrix with value x+y at position (x, y) */
void init_matrix(struct vec_f2d *m) {
  for (int64_t y = 0; y < m->sizes[1]; y++)
    for (int64_t x = 0; x < m->sizes[0]; x++)
      vec_f2d_set(m, x, y, 1 + x * y);
}

void die_usage(const char *program_name) {
  fprintf(stderr, "Usage: %s [-v]\n", program_name);
  exit(1);
}

int main(int argc, char **argv) {
  struct vec_f2d a;
  struct vec_f1d x, x_ref, b;
  int verbose = 0;
  int n = N;

  if (vec_f2d_alloc(&a, n, n) || vec_f1d_alloc(&x, n) ||
      vec_f1d_alloc(&x_ref, n) || vec_f1d_alloc(&b, n)) {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_vector(&x);
  init_vector(&x_ref);
  init_vector(&b);

  scop_entry(VEC2D_ARGS(&a), VEC1D_ARGS(&b), VEC1D_ARGS(&x));

  mm_refimpl(&a, &x_ref, &b);

  if (verbose) {
    puts("Result O:");
    vec_f1d_dump(&x);
    puts("");

    puts("Reference O:");
    vec_f1d_dump(&x_ref);
    puts("");
  }

  if (!vec_f1d_compare(&x, &x_ref)) {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&a);

  vec_f1d_destroy(&x);
  vec_f1d_destroy(&x_ref);

  return 0;
}
