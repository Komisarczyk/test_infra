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
extern void threemm(
    float *a_allocatedptr, float *a_alignedptr,
    int64_t a_offset, int64_t a_sizes0, int64_t a_sizes1,
    int64_t a_strides0, int64_t a_strides1,

    float *b_allocatedptr, float *b_alignedptr,
    int64_t b_offset, int64_t b_sizes0, int64_t b_sizes1,
    int64_t b_strides0, int64_t b_strides1,

    float *c_allocatedptr, float *c_alignedptr,
    int64_t c_offset, int64_t c_sizes0, int64_t c_sizes1,
    int64_t c_strides0, int64_t c_strides1,

    float *d_allocatedptr, float *d_alignedptr,
    int64_t d_offset, int64_t d_sizes0, int64_t d_sizes1,
    int64_t d_strides0, int64_t d_strides1,

    float *e_allocatedptr, float *e_alignedptr,
    int64_t e_offset, int64_t e_sizes0, int64_t e_sizes1,
    int64_t e_strides0, int64_t e_strides1,

    float *f_allocatedptr, float *f_alignedptr,
    int64_t f_offset, int64_t f_sizes0, int64_t f_sizes1,
    int64_t f_strides0, int64_t f_strides1,

    float *g_allocatedptr, float *g_alignedptr,
    int64_t g_offset, int64_t g_sizes0, int64_t g_sizes1,
    int64_t g_strides0, int64_t g_strides1

);

/* Reference implementation of a matrix multiplication */
void mm_refimpl(struct vec_f2d *a, struct vec_f2d *b, struct vec_f2d *c, struct vec_f2d *d, struct vec_f2d *e, struct vec_f2d *f, struct vec_f2d *g)
{

  for (int i = 0; i < e->sizes[0]; i++)
    for (int j = 0; j < e->sizes[1]; j++)
    {
      vec_f2d_set(e, i, j, 0.0f);
      for (int k = 0; k < a->sizes[1]; ++k)
        vec_f2d_set(e, i, j, vec_f2d_get(e, i, j) + vec_f2d_get(a, i, k) * vec_f2d_get(b, k, j));
    }
  /* F := C*D */
  for (int i = 0; i < f->sizes[0]; i++)
    for (int j = 0; j < f->sizes[1]; j++)
    {
      vec_f2d_set(f, i, j, 0.0f);
      for (int k = 0; k < c->sizes[1]; ++k)
        vec_f2d_set(f, i, j, vec_f2d_get(f, i, j) + vec_f2d_get(c, i, k) * vec_f2d_get(d, k, j));
    }
  /* G := E*F */
  for (int i = 0; i < g->sizes[0]; i++)
    for (int j = 0; j < g->sizes[1]; j++)
    {
      vec_f2d_set(g, i, j, 0.0f);
      for (int k = 0; k < e->sizes[1]; ++k)
        vec_f2d_set(g, i, j, vec_f2d_get(g, i, j) + vec_f2d_get(e, i, k) * vec_f2d_get(f, k, j));
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
  struct vec_f2d a, b, c, d, e, f, g, g_ref;
  int verbose = 1;
  int n = 8;
  int m = 9;
  int p = 10;
  int r = 11;
  int q = 12;

  if (vec_f2d_alloc(&a, n, p) || vec_f2d_alloc(&b, p, m) || vec_f2d_alloc(&c, m, q) || vec_f2d_alloc(&d, q, r) || vec_f2d_alloc(&e, n, m) || vec_f2d_alloc(&f, m, r) || vec_f2d_alloc(&g, n, r) || vec_f2d_alloc(&g_ref, n, r))
  {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_matrix(&b);
  init_matrix(&c);
  init_matrix(&d);
  init_matrix(&e);
  init_matrix(&f);
  init_matrix(&g);
  init_matrix(&g_ref);

  if (verbose)
  {
    puts("O_REF:");
    vec_f2d_dump(&g_ref);
    puts("");

    puts("O:");
    vec_f2d_dump(&g);
    puts("");
  }

  threemm(VEC2D_ARGS(&a), VEC2D_ARGS(&b), VEC2D_ARGS(&c), VEC2D_ARGS(&d), VEC2D_ARGS(&e), VEC2D_ARGS(&f), VEC2D_ARGS(&g));

  mm_refimpl(&a, &b, &c, &d, &e, &f, &g_ref);

  if (verbose)
  {
    puts("Result O:");
    vec_f2d_dump(&g);
    puts("");

    puts("Reference O:");
    vec_f2d_dump(&g_ref);
    puts("");
  }

  if (!vec_f2d_compare(&g, &g_ref))
  {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&a);
  vec_f2d_destroy(&b);
  vec_f2d_destroy(&c);
  vec_f2d_destroy(&d);
  vec_f2d_destroy(&e);
  vec_f2d_destroy(&f);
  vec_f2d_destroy(&g);
  vec_f2d_destroy(&g_ref);

  return 0;
}
