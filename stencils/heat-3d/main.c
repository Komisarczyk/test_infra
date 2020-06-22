#define ALPHA 1.0f
#define BETA 2.0f
#include "../../../polybench/stencils/heat-3d/heat-3d.h"
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
                       int64_t a_sizes2, int64_t a_strides0, int64_t a_strides1,
                       int64_t a_strides2,

                       float *b_allocatedptr, float *b_alignedptr,
                       int64_t b_offset, int64_t b_sizes0, int64_t b_sizes1,
                       int64_t b_sizes2, int64_t b_strides0, int64_t b_strides1,
                       int64_t b_strides2);

/* Reference implementation of a matrix multiplication */
void mm_refimpl(struct vec_f3d *a, struct vec_f3d *b) {

  for (int t = 1; t <= TSTEPS; t++) {
    for (int i = 1; i < b->sizes[0] - 1; i++) {
      for (int j = 1; j < b->sizes[0] - 1; j++) {
        for (int k = 1; k < b->sizes[0] - 1; k++) {
          vec_f3d_set(b, i, j, k,
                      (0.125f) * (vec_f3d_get(a, i + 1, j, k) -
                                  2.0f * vec_f3d_get(a, i, j, k) +
                                  vec_f3d_get(a, i - 1, j, k)) +
                          (0.125f) * (vec_f3d_get(a, i, j + 1, k) -
                                      2.0f * vec_f3d_get(a, i, j, k) +
                                      vec_f3d_get(a, i, j - 1, k)) +
                          (0.125f) * (vec_f3d_get(a, i, j, k + 1) -
                                      2.0f * vec_f3d_get(a, i, j, k) +
                                      vec_f3d_get(a, i, j, k - 1)) +
                          vec_f3d_get(a, i, j, k));
        }
      }
    }
    for (int i = 1; i < b->sizes[0] - 1; i++) {
      for (int j = 1; j < b->sizes[0] - 1; j++) {
        for (int k = 1; k < b->sizes[0] - 1; k++) {
          vec_f3d_set(a, i, j, k,
                      (0.125f) * (vec_f3d_get(b, i + 1, j, k) -
                                  2.0f * vec_f3d_get(b, i, j, k) +
                                  vec_f3d_get(b, i - 1, j, k)) +
                          (0.125f) * (vec_f3d_get(b, i, j + 1, k) -
                                      2.0f * vec_f3d_get(b, i, j, k) +
                                      vec_f3d_get(b, i, j - 1, k)) +
                          (0.125f) * (vec_f3d_get(b, i, j, k + 1) -
                                      2.0f * vec_f3d_get(b, i, j, k) +
                                      vec_f3d_get(b, i, j, k - 1)) +
                          vec_f3d_get(b, i, j, k));
        }
      }
    }
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
/* Initialize matrix with value x+y at position (x, y) */
void init_matrix3d(struct vec_f3d *m) {
  for (int64_t z = 0; z < m->sizes[2]; z++)
    for (int64_t y = 0; y < m->sizes[1]; y++)
      for (int64_t x = 0; x < m->sizes[0]; x++)
        vec_f3d_set(m, x, y, z, z + 1 + x * y);
}

void die_usage(const char *program_name) {
  fprintf(stderr, "Usage: %s [-v]\n", program_name);
  exit(1);
}

int main(int argc, char **argv) {
  struct vec_f3d a, b, a_ref;
  int verbose = 0;
  int n = N;

  if (vec_f3d_alloc(&a, n, n, n) || vec_f3d_alloc(&a_ref, n, n, n) ||
      vec_f3d_alloc(&b, n, n, n)) {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix3d(&a);
  init_matrix3d(&b);
  init_matrix3d(&a_ref);

  if (verbose) {
    puts("Result O:");
    vec_f3d_dump(&a);
    puts("");

    puts("Reference O:");
    vec_f3d_dump(&a_ref);
    puts("");
  }

  scop_entry(VEC3D_ARGS(&a), VEC3D_ARGS(&b));

  init_matrix3d(&b);
  mm_refimpl(&a_ref, &b);

  if (verbose) {
    puts("Result O:");
    vec_f3d_dump(&a);
    puts("");

    puts("Reference O:");
    vec_f3d_dump(&a_ref);
    puts("");
  }

  if (!vec_f3d_compare(&a, &a_ref)) {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f3d_destroy(&a);

  vec_f3d_destroy(&b);
  vec_f3d_destroy(&a_ref);

  return 0;
}
