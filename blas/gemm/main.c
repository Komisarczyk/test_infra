#include "../../../polybench-c-4.2.1-beta/linear-algebra/blas/gemm/gemm.h"
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
extern void scop_entry(const float *a_allocatedptr, const float *a_alignedptr,
                       int64_t a_offset, int64_t a_sizes0, int64_t a_sizes1,
                       int64_t a_strides0, int64_t a_strides1,
                       const float *b_allocatedptr, const float *b_alignedptr,
                       int64_t b_offset, int64_t b_sizes0, int64_t b_sizes1,
                       int64_t b_strides0, int64_t b_strides1,
                       float *o_allocatedptr, float *o_alignedptr,
                       int64_t o_offset, int64_t o_sizes0, int64_t o_sizes1,
                       int64_t o_strides0, int64_t o_strides1,
                       const float alpha, const float beta);

/* Reference implementation of a matrix multiplication */
void mm_refimpl(const struct vec_f2d *a, const struct vec_f2d *b,
                struct vec_f2d *c) {

  float beta = BETA;
  float alpha = ALPHA;

  for (int64_t i = 0; i < c->sizes[0]; i++) {
    for (int64_t j = 0; j < c->sizes[1]; j++)
      vec_f2d_set(c, i, j, vec_f2d_get(c, i, j) * beta);
    for (int64_t k = 0; k < a->sizes[1]; k++) {
      for (int64_t j = 0; j < c->sizes[1]; j++)
        vec_f2d_set(c, i, j,
                    vec_f2d_get(c, i, j) +
                        alpha * vec_f2d_get(a, i, k) * vec_f2d_get(b, k, j));
    }
  }
}

/* Initialize matrix with value x+y at position (x, y) */
void init_matrix(struct vec_f2d *m) {
  for (int64_t y = 0; y < m->sizes[1]; y++)
    for (int64_t x = 0; x < m->sizes[0]; x++)
      vec_f2d_set(m, x, y, x + 2 * y);
}

void die_usage(const char *program_name) {
  fprintf(stderr, "Usage: %s [-v]\n", program_name);
  exit(1);
}

int main(int argc, char **argv) {
  struct vec_f2d a, b, o, o_ref;
  int verbose = 0;
  int n = NI;
  int m = NJ;
  int k = NK;
  if (vec_f2d_alloc(&a, n, k) || vec_f2d_alloc(&b, k, m) ||
      vec_f2d_alloc(&o, n, m) || vec_f2d_alloc(&o_ref, n, m)) {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_matrix(&b);

  if (verbose) {
    puts("B:");
    vec_f2d_dump(&b);
    puts("");

    puts("O:");
    vec_f2d_dump(&o);
    puts("");
  }

  scop_entry(VEC2D_ARGS(&a), VEC2D_ARGS(&b), VEC2D_ARGS(&o), ALPHA, BETA);
  mm_refimpl(&a, &b, &o_ref);

  if (verbose) {
    puts("Result O:");
    vec_f2d_dump(&o);
    puts("");

    puts("Reference O:");
    vec_f2d_dump(&o_ref);
    puts("");
  }

  if (!vec_f2d_compare(&o, &o_ref)) {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&a);
  vec_f2d_destroy(&b);
  vec_f2d_destroy(&o);
  vec_f2d_destroy(&o_ref);

  return 0;
}
