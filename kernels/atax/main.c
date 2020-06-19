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
extern void atax(
    float *a_allocatedptr, float *a_alignedptr,
    int64_t a_offset, int64_t a_sizes0, int64_t a_sizes1,
    int64_t a_strides0, int64_t a_strides1,

    float *tmp_allocatedptr, float *tmp_alignedptr,
    int64_t tmp_offset, int64_t tmp_sizes0, int64_t tmp_strides0,

    float *x_allocatedptr, float *x_alignedptr,
    int64_t x_offset, int64_t x_sizes0, int64_t x_strides0,

    float *y_allocatedptr, float *y_alignedptr,
    int64_t y_offset, int64_t y_sizes0, int64_t y_strides0);

/* Reference implementation of a matrix multiplication */
void mm_refimpl(struct vec_f2d *a, struct vec_f1d *tmp, struct vec_f1d *x, struct vec_f1d *y_ref)
{

  for (int i = 0; i < y_ref->sizes[0]; i++)
    vec_f1d_set(y_ref, i, 0.0);
  for (int i = 0; i < tmp->sizes[0]; i++)
  {
    vec_f1d_set(tmp, i, 0.0);
    for (int j = 0; j < x->sizes[0]; j++)
      vec_f1d_set(tmp, i, vec_f2d_get(a, i, j) * vec_f1d_get(x, j) + vec_f1d_get(tmp, i));
    for (int j = 0; j < y_ref->sizes[0]; j++)
      vec_f1d_set(y_ref, j, vec_f1d_get(tmp, i) * vec_f2d_get(a, i, j) + vec_f1d_get(y_ref, j));
  }
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
  struct vec_f1d tmp, x, y, y_ref;
  int verbose = 1;
  int n = 19;
  int m = 21;

  if (vec_f2d_alloc(&a, n, m) || vec_f1d_alloc(&tmp, n) ||
      vec_f1d_alloc(&x, m) || vec_f1d_alloc(&y, m) || vec_f1d_alloc(&y_ref, m))
  {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_vector(&tmp);
  init_vector(&x);
  init_vector(&y);
  init_vector(&y_ref);

  if (verbose)
  {
    puts("A:");
    vec_f2d_dump(&a);
    puts("");

    puts("O:");
    vec_f1d_dump(&y);
    puts("");
  }

  atax(VEC2D_ARGS(&a), VEC1D_ARGS(&tmp), VEC1D_ARGS(&x), VEC1D_ARGS(&y));

  mm_refimpl(&a, &tmp, &x, &y_ref);

  if (verbose)
  {
    puts("Result O:");
    vec_f1d_dump(&y);
    puts("");

    puts("Reference O:");
    vec_f1d_dump(&y_ref);
    puts("");
  }

  if (!vec_f1d_compare(&y, &y_ref))
  {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&a);

  vec_f1d_destroy(&tmp);
  vec_f1d_destroy(&x);
  vec_f1d_destroy(&y);
  vec_f1d_destroy(&y_ref);

  return 0;
}
