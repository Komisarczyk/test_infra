#define ALPHA 1.0f
#define BETA 2.0f
#include "../../../polybench/linear-algebra/blas/trmm/trmm.h"
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

                       float *b_allocatedptr, float *b_alignedptr,
                       int64_t b_offset, int64_t b_sizes0, int64_t b_sizes1,
                       int64_t b_strides0, int64_t b_strides1,

                       const float alpha);

/* Reference implementation of a matrix multiplication */
void mm_refimpl(struct vec_f2d *a, struct vec_f2d *b) {

  float alpha = ALPHA;

  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      for (int k = i; k < M; k++)
        vec_f2d_set(b, i, j, vec_f2d_get(b, i, j) +
                                 vec_f2d_get(a, k, i) * vec_f2d_get(b, k, j));
      vec_f2d_set(b, i, j, alpha * vec_f2d_get(b, i, j));
    }
}

/* Initialize vector with value x at position (x) */
void init_vector(struct vec_f1d *v) {
  for (int64_t x = 0; x < v->sizes[0]; x++)
    vec_f1d_set(v, x, x);
}

/* Initialize matrix with value x+y at position (x, y) */
void init_matrix(struct vec_f2d *m) {
  for (int64_t x = 0; x < m->sizes[0]; x++)
    for (int64_t y = 0; y < m->sizes[1]; y++)

      vec_f2d_set(m, x, y, 2);
}

void die_usage(const char *program_name) {
  fprintf(stderr, "Usage: %s [-v]\n", program_name);
  exit(1);
}

int main(int argc, char **argv) {
  struct vec_f2d a, c, d;
  int verbose = 0;
  int n = M;
  int m = N;

  if (vec_f2d_alloc(&a, n, n) || vec_f2d_alloc(&c, n, m) ||
      vec_f2d_alloc(&d, n, m)) {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_matrix(&c);
  init_matrix(&d);

  if (verbose) {
    puts("O_REF:");
    vec_f2d_dump(&d);
    puts("");

    puts("O:");
    vec_f2d_dump(&c);
    puts("");
  }

  scop_entry(VEC2D_ARGS(&a), VEC2D_ARGS(&c), ALPHA);

  mm_refimpl(&a, &d);

  if (verbose) {
    puts("Result O:");
    vec_f2d_dump(&c);
    puts("");

    puts("Reference O:");
    vec_f2d_dump(&d);
    puts("");
  }

  if (!vec_f2d_compare(&c, &d)) {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&a);

  vec_f2d_destroy(&c);
  vec_f2d_destroy(&d);

  return 0;
}
