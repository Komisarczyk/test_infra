#include "../../../polybench-c-4.2.1-beta/linear-algebra/kernels/2mm/2mm.h"
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
extern void scop_entry(
    float *a_allocatedptr, float *a_alignedptr, int64_t a_offset,
    int64_t a_sizes0, int64_t a_sizes1, int64_t a_strides0, int64_t a_strides1,

    float *b_allocatedptr, float *b_alignedptr, int64_t b_offset,
    int64_t b_sizes0, int64_t b_sizes1, int64_t b_strides0, int64_t b_strides1,

    float *c_allocatedptr, float *c_alignedptr, int64_t c_offset,
    int64_t c_sizes0, int64_t c_sizes1, int64_t c_strides0, int64_t c_strides1,

    float *d_allocatedptr, float *d_alignedptr, int64_t d_offset,
    int64_t d_sizes0, int64_t d_sizes1, int64_t d_strides0, int64_t d_strides1,

    const float alpha, const float beta,

    float *tmp_allocatedptr, float *tmp_alignedptr, int64_t tmp_offset,
    int64_t tmp_sizes0, int64_t tmp_sizes1, int64_t tmp_strides0,
    int64_t tmp_strides1

    );

/* Reference implementation of a matrix multiplication */
void mm_refimpl(struct vec_f2d *a, struct vec_f2d *b, struct vec_f2d *c,
                struct vec_f2d *d, struct vec_f2d *tmp) {

  float alpha = ALPHA;
  float beta = BETA;

  for (int i = 0; i < tmp->sizes[0]; i++)
    for (int j = 0; j < tmp->sizes[1]; j++) {
      vec_f2d_set(tmp, i, j, 0.0f);
      for (int k = 0; k < a->sizes[1]; ++k)
        vec_f2d_set(tmp, i, j,
                    vec_f2d_get(tmp, i, j) +
                        alpha * vec_f2d_get(a, i, k) * vec_f2d_get(b, k, j));
    }
  for (int i = 0; i < d->sizes[0]; i++)
    for (int j = 0; j < d->sizes[1]; j++) {
      vec_f2d_set(d, i, j, vec_f2d_get(d, i, j) * beta);
      for (int k = 0; k < tmp->sizes[1]; ++k)
        vec_f2d_set(d, i, j, vec_f2d_get(d, i, j) +
                                 vec_f2d_get(tmp, i, k) * vec_f2d_get(c, k, j));
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
  struct vec_f2d a, b, c, d, d_ref, tmp;
  int verbose = 0;
  int n = NI;
  int m = NJ;
  int p = NK;
  int r = NL;

  if (vec_f2d_alloc(&a, n, p) || vec_f2d_alloc(&b, p, m) ||
      vec_f2d_alloc(&c, m, r) || vec_f2d_alloc(&d, n, r) ||
      vec_f2d_alloc(&d_ref, n, r) || vec_f2d_alloc(&tmp, n, m)) {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_matrix(&b);
  init_matrix(&c);
  init_matrix(&d);
  init_matrix(&d_ref);
  init_matrix(&tmp);

  if (verbose) {
    puts("O_REF:");
    vec_f2d_dump(&d_ref);
    puts("");

    puts("O:");
    vec_f2d_dump(&d);
    puts("");
  }

  scop_entry(VEC2D_ARGS(&a), VEC2D_ARGS(&b), VEC2D_ARGS(&c), VEC2D_ARGS(&d),
             ALPHA, BETA, VEC2D_ARGS(&tmp));

  init_matrix(&tmp);

  mm_refimpl(&a, &b, &c, &d_ref, &tmp);

  if (verbose) {
    puts("Result O:");
    vec_f2d_dump(&d);
    puts("");

    puts("Reference O:");
    vec_f2d_dump(&d_ref);
    puts("");
  }

  if (!vec_f2d_compare(&d, &d_ref)) {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&a);
  vec_f2d_destroy(&b);
  vec_f2d_destroy(&c);
  vec_f2d_destroy(&d);
  vec_f2d_destroy(&d_ref);
  vec_f2d_destroy(&tmp);

  return 0;
}
