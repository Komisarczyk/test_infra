#include "../../../polybench-c-4.2.1-beta/stencils/jacobi-1d/jacobi-1d.h"
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
extern void scop_entry(float *p_allocatedptr, float *p_alignedptr,
                       int64_t p_offset, int64_t p_sizes0, int64_t p_strides0,

                       float *r_allocatedptr, float *r_alignedptr,
                       int64_t r_offset, int64_t r_sizes0, int64_t r_strides0);

/* Reference implementation of a matrix multiplication */
void mm_refimpl(struct vec_f1d *a, struct vec_f1d *b) {

  for (int t = 0; t < TSTEPS; t++) {
    for (int i = 1; i < b->sizes[0] - 1; i++)

      vec_f1d_set(b, i, 0.33333f * (vec_f1d_get(a, i - 1) + vec_f1d_get(a, i) +
                                    vec_f1d_get(a, i + 1)));
    for (int i = 1; i < a->sizes[0] - 1; i++)
      vec_f1d_set(a, i, 0.33333f * (vec_f1d_get(b, i - 1) + vec_f1d_get(b, i) +
                                    vec_f1d_get(b, i + 1)));
  }
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
  struct vec_f1d a, a_ref, b;
  int verbose = 0;
  int n = N;

  if (vec_f1d_alloc(&a, n) || vec_f1d_alloc(&a_ref, n) ||
      vec_f1d_alloc(&b, n)) {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_vector(&a);
  init_vector(&a_ref);
  init_vector(&b);

  if (verbose) {
    puts("Result O:");
    vec_f1d_dump(&a);
    puts("");

    puts("Reference O:");
    vec_f1d_dump(&a_ref);
    puts("");
  }

  scop_entry(VEC1D_ARGS(&a), VEC1D_ARGS(&b));

  init_vector(&b);
  mm_refimpl(&a_ref, &b);

  if (verbose) {
    puts("Result O:");
    vec_f1d_dump(&a);
    puts("");

    puts("Reference O:");
    vec_f1d_dump(&a_ref);
    puts("");
  }

  if (!vec_f1d_compare(&a, &a_ref)) {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f1d_destroy(&a);

  vec_f1d_destroy(&b);
  vec_f1d_destroy(&a_ref);

  return 0;
}
