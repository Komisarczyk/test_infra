#define ALPHA 1.0f
#define BETA 2.0f
#include "../../../polybench/linear-algebra/blas/gemver/gemver.h"
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
extern void
scop_entry(float *a_allocatedptr, float *a_alignedptr, int64_t a_offset,
           int64_t a_sizes0, int64_t a_sizes1, int64_t a_strides0,
           int64_t a_strides1,

           const float alpha, const float beta,

           float *u1_allocatedptr, float *u1_alignedptr, int64_t u1_offset,
           int64_t u1_sizes0, int64_t u1_strides0,

           float *v1_allocatedptr, float *v1_alignedptr, int64_t v1_offset,
           int64_t v1_sizes0, int64_t v1_strides0,

           float *u2_allocatedptr, float *u2_alignedptr, int64_t u2_offset,
           int64_t u2_sizes0, int64_t u2_strides0,

           float *v2_allocatedptr, float *v2_alignedptr, int64_t v2_offset,
           int64_t v2_sizes0, int64_t v2_strides0,

           float *w_allocatedptr, float *w_alignedptr, int64_t w_offset,
           int64_t w_sizes0, int64_t w_strides0,

           float *x_allocatedptr, float *x_alignedptr, int64_t x_offset,
           int64_t x_sizes0, int64_t x_strides0,

           float *y_allocatedptr, float *y_alignedptr, int64_t y_offset,
           int64_t y_sizes0, int64_t y_strides0,

           float *z_allocatedptr, const float *z_alignedptr, int64_t z_offset,
           int64_t z_sizes0, int64_t z_strides0);

/* Reference implementation of a matrix multiplication */
void mm_refimpl(struct vec_f2d *a, struct vec_f1d *u1, struct vec_f1d *v1,
                struct vec_f1d *u2, struct vec_f1d *v2, struct vec_f1d *x,
                struct vec_f1d *y, struct vec_f1d *z, struct vec_f1d *w_ref) {

  float alpha = ALPHA;
  float beta = BETA;

  for (int i = 0; i < a->sizes[0]; i++)
    for (int j = 0; j < a->sizes[1]; j++)
      vec_f2d_set(a, i, j, vec_f2d_get(a, i, j) +
                               vec_f1d_get(u1, i) * vec_f1d_get(v1, j) +
                               vec_f1d_get(u2, i) * vec_f1d_get(v2, j));

  for (int i = 0; i < x->sizes[0]; i++)
    for (int j = 0; j < a->sizes[0]; j++)
      vec_f1d_set(x, i, vec_f1d_get(x, i) +
                            beta * vec_f2d_get(a, j, i) * vec_f1d_get(y, j));

  for (int i = 0; i < x->sizes[0]; i++)
    vec_f1d_set(x, i, vec_f1d_get(x, i) + vec_f1d_get(z, i));

  for (int i = 0; i < w_ref->sizes[0]; i++)
    for (int j = 0; j < a->sizes[1]; j++) {
      vec_f1d_set(w_ref, i,
                  vec_f1d_get(w_ref, i) +
                      alpha * vec_f2d_get(a, i, j) * vec_f1d_get(x, j));
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
  struct vec_f2d a;
  struct vec_f1d u1, v1, u2, v2, w, w_ref, x, y, z;
  int verbose = 0;
  int n = N;

  if (vec_f2d_alloc(&a, n, n) || vec_f1d_alloc(&u1, n) ||
      vec_f1d_alloc(&v1, n) || vec_f1d_alloc(&u2, n) || vec_f1d_alloc(&v2, n) ||
      vec_f1d_alloc(&x, n) || vec_f1d_alloc(&y, n) || vec_f1d_alloc(&z, n) ||
      vec_f1d_alloc(&w, n) || vec_f1d_alloc(&w_ref, n)) {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_vector(&u1);
  init_vector(&v1);
  init_vector(&u2);
  init_vector(&v2);
  init_vector(&x);
  init_vector(&y);
  init_vector(&z);
  init_vector(&w);
  init_vector(&w_ref);

  if (verbose) {
    puts("A:");
    vec_f2d_dump(&a);
    puts("");

    puts("O:");
    vec_f1d_dump(&w);
    puts("");
  }

  scop_entry(VEC2D_ARGS(&a), ALPHA, BETA, VEC1D_ARGS(&u1), VEC1D_ARGS(&v1),
             VEC1D_ARGS(&u2), VEC1D_ARGS(&v2), VEC1D_ARGS(&w), VEC1D_ARGS(&x),
             VEC1D_ARGS(&y), VEC1D_ARGS(&z));

  init_matrix(&a);
  init_vector(&u1);
  init_vector(&v1);
  init_vector(&u2);
  init_vector(&v2);
  init_vector(&x);
  init_vector(&y);
  init_vector(&z);

  mm_refimpl(&a, &u1, &v1, &u2, &v2, &x, &y, &z, &w_ref);

  if (verbose) {
    puts("Result O:");
    vec_f1d_dump(&w);
    puts("");

    puts("Reference O:");
    vec_f1d_dump(&w_ref);
    puts("");
  }

  if (!vec_f1d_compare(&w, &w_ref)) {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&a);
  vec_f1d_destroy(&u1);
  vec_f1d_destroy(&v1);
  vec_f1d_destroy(&u2);
  vec_f1d_destroy(&v2);
  vec_f1d_destroy(&x);
  vec_f1d_destroy(&y);
  vec_f1d_destroy(&z);
  vec_f1d_destroy(&w);
  vec_f1d_destroy(&w_ref);

  return 0;
}
