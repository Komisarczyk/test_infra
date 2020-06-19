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
// See: https://mlir.llvm.org/docs/ConversionToLLVMDialect/#calling-convention-for-memref
extern void mvt(
    float *a_allocatedptr, float *a_alignedptr,
    int64_t a_offset, int64_t a_sizes0, int64_t a_sizes1,
    int64_t a_strides0, int64_t a_strides1,

    float *tmp_allocatedptr, float *tmp_alignedptr,
    int64_t tmp_offset, int64_t tmp_sizes0, int64_t tmp_strides0,

    float *x_allocatedptr, float *x_alignedptr,
    int64_t x_offset, int64_t x_sizes0, int64_t x_strides0,

    float *y_allocatedptr, float *y_alignedptr,
    int64_t y_offset, int64_t y_sizes0, int64_t y_strides0,

    float *z_allocatedptr, float *z_alignedptr,
    int64_t z_offset, int64_t z_sizes0, int64_t z_strides0);

/* Reference implementation of a matrix multiplication */
void mm_refimpl(struct vec_f2d *a, struct vec_f1d *x1, struct vec_f1d *y1, struct vec_f1d *x2, struct vec_f1d *y2)
{

  for (int i = 0; i < x1->sizes[0]; i++)
    for (int j = 0; j < a->sizes[1]; j++)
      vec_f1d_set(x1, i, vec_f2d_get(a, i, j) * vec_f1d_get(y1, j) + vec_f1d_get(x1, i));
  for (int i = 0; i < x2->sizes[0]; i++)
    for (int j = 0; j < a->sizes[0]; j++)
      vec_f1d_set(x2, i, vec_f2d_get(a, j, i) * vec_f1d_get(y2, j) + vec_f1d_get(x2, i));
}

/* Initialize vector with value x at position (x) */
void init_vector(struct vec_f1d *v)
{
  for (int64_t x = 0; x < v->sizes[0]; x++)
    vec_f1d_set(v, x, x);
}

/* Initialize matrix with value x+y at position (x, y) */
void init_matrix(struct vec_f2d *m)
{
  for (int64_t y = 0; y < m->sizes[1]; y++)
    for (int64_t x = 0; x < m->sizes[0]; x++)
      vec_f2d_set(m, x, y, x + y);
}

void die_usage(const char *program_name)
{
  fprintf(stderr, "Usage: %s [-v]\n", program_name);
  exit(1);
}

int main(int argc, char **argv)
{
  struct vec_f2d a;
  struct vec_f1d tmp, x1, y1, x2, y2, x2_ref;
  int verbose = 1;
  int n = 12;

  if (vec_f2d_alloc(&a, n, n) || vec_f1d_alloc(&x1, n) ||
      vec_f1d_alloc(&x2, n) || vec_f1d_alloc(&y1, n) || vec_f1d_alloc(&y2, n) || vec_f1d_alloc(&x2_ref, n))
  {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_vector(&tmp);
  init_vector(&x1);
  init_vector(&y1);
  init_vector(&x2);
  init_vector(&y2);
  init_vector(&x2_ref);

  if (verbose)
  {
    puts("A:");
    vec_f2d_dump(&a);
    puts("");

    puts("O:");
    vec_f1d_dump(&y1);
    puts("");
  }

  mvt(VEC2D_ARGS(&a), VEC1D_ARGS(&x1), VEC1D_ARGS(&x2), VEC1D_ARGS(&y1), VEC1D_ARGS(&y2));

  mm_refimpl(&a, &x1, &y1, &x2_ref, &y2);

  if (verbose)
  {
    puts("Result O:");
    vec_f1d_dump(&x2);
    puts("");

    puts("Reference O:");
    vec_f1d_dump(&x2_ref);
    puts("");
  }

  if (!vec_f1d_compare(&x2, &x2_ref))
  {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&a);

  vec_f1d_destroy(&x1);
  vec_f1d_destroy(&y1);
  vec_f1d_destroy(&x2);
  vec_f1d_destroy(&y2);
  vec_f1d_destroy(&x2_ref);

  return 0;
}
