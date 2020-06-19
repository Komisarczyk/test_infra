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
extern void jacobitwod(
    float *a_allocatedptr, float *a_alignedptr,
    int64_t a_offset, int64_t a_sizes0, int64_t a_sizes1,
    int64_t a_strides0, int64_t a_strides1,

    float *b_allocatedptr, float *b_alignedptr,
    int64_t b_offset, int64_t b_sizes0, int64_t b_sizes1,
    int64_t b_strides0, int64_t b_strides1);

/* Reference implementation of a matrix multiplication */
void mm_refimpl(struct vec_f2d *a, struct vec_f2d *b)
{

  for (int t = 0; t < 5; t++)
  {
    for (int i = 1; i < b->sizes[0] - 1; i++)
      for (int j = 1; j < b->sizes[1] - 1; j++)
        //B[i][j] = (0.2f) * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
        vec_f2d_set(b, i, j, 0.2f * (vec_f2d_get(a, i, j) + vec_f2d_get(a, i, j - 1) + vec_f2d_get(a, i, j + 1) + vec_f2d_get(a, i + 1, j) + vec_f2d_get(a, i - 1, j)));
    for (int i = 1; i < a->sizes[0] - 1; i++)
      for (int j = 1; j < b->sizes[1] - 1; j++)
        //A[i][j] = (0.2f) * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
        vec_f2d_set(a, i, j, 0.2f * (vec_f2d_get(b, i, j) + vec_f2d_get(b, i, j - 1) + vec_f2d_get(b, i, j + 1) + vec_f2d_get(b, i + 1, j) + vec_f2d_get(b, i - 1, j)));
  }
}

/* Initialize vector with value x at position (x) */
void init_vector(struct vec_f1d *v)
{
  for (int64_t x = 0; x < v->sizes[0]; x++)
    vec_f1d_set(v, x, 1 + x * x);
}

/* Initialize matrix with value x+y at position (x, y) */
void init_matrix(struct vec_f2d *m)
{
  for (int64_t y = 0; y < m->sizes[1]; y++)
    for (int64_t x = 0; x < m->sizes[0]; x++)
      vec_f2d_set(m, x, y, 1 + x * y);
}

void die_usage(const char *program_name)
{
  fprintf(stderr, "Usage: %s [-v]\n", program_name);
  exit(1);
}

int main(int argc, char **argv)
{
  struct vec_f2d a, a_ref, b;
  int verbose = 1;
  int n = 13;

  if (vec_f2d_alloc(&a, n, n) || vec_f2d_alloc(&a_ref, n, n) || vec_f2d_alloc(&b, n, n))
  {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_matrix(&a_ref);
  init_matrix(&b);

  jacobitwod(VEC2D_ARGS(&a), VEC2D_ARGS(&b));

  init_matrix(&b);
  mm_refimpl(&a_ref, &b);

  if (verbose)
  {
    puts("Result O:");
    vec_f2d_dump(&a);
    puts("");

    puts("Reference O:");
    vec_f2d_dump(&a_ref);
    puts("");
  }

  if (!vec_f2d_compare(&a, &a_ref))
  {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&a);
  vec_f2d_destroy(&b);
  vec_f2d_destroy(&a_ref);

  return 0;
}
