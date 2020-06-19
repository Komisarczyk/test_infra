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
extern void bicg(
    float *a_allocatedptr, float *a_alignedptr,
    int64_t a_offset, int64_t a_sizes0, int64_t a_sizes1,
    int64_t a_strides0, int64_t a_strides1,

    float *s_allocatedptr, float *s_alignedptr,
    int64_t s_offset, int64_t s_sizes0, int64_t s_strides0,

    float *q_allocatedptr, float *q_alignedptr,
    int64_t q_offset, int64_t q_sizes0, int64_t q_strides0,

    float *p_allocatedptr, float *p_alignedptr,
    int64_t p_offset, int64_t p_sizes0, int64_t p_strides0,

    float *r_allocatedptr, float *r_alignedptr,
    int64_t r_offset, int64_t r_sizes0, int64_t r_strides0);

/* Reference implementation of a matrix multiplication */
void mm_refimpl(struct vec_f2d *a, struct vec_f1d *s, struct vec_f1d *q, struct vec_f1d *p, struct vec_f1d *r)
{

  for (int i = 0; i < s->sizes[0]; i++)
    vec_f1d_set(s, i, 0.0f);
  for (int i = 0; i < q->sizes[0]; i++)
  {
    vec_f1d_set(q, i, 0.0f);
    for (int j = 0; j < s->sizes[0]; j++)
    {
      vec_f1d_set(s, j, vec_f2d_get(a, i, j) * vec_f1d_get(r, i) + vec_f1d_get(s, j));
      vec_f1d_set(q, i, vec_f2d_get(a, i, j) * vec_f1d_get(p, j) + vec_f1d_get(q, i));
    }
  }
}

/* Initialize vector with value x at position (x) */
void init_vector(struct vec_f1d *v)
{
  for (int64_t x = 0; x < v->sizes[0]; x++)
    vec_f1d_set(v, x, x * x);
}

/* Initialize matrix with value x+y at position (x, y) */
void init_matrix(struct vec_f2d *m)
{
  for (int64_t y = 0; y < m->sizes[1]; y++)
    for (int64_t x = 0; x < m->sizes[0]; x++)
      vec_f2d_set(m, x, y, x * y);
}

void die_usage(const char *program_name)
{
  fprintf(stderr, "Usage: %s [-v]\n", program_name);
  exit(1);
}

int main(int argc, char **argv)
{
  struct vec_f2d a;
  struct vec_f1d s, q, p, r, q_ref;
  int verbose = 1;
  int n = 19;
  int m = 21;

  if (vec_f2d_alloc(&a, m, n) || vec_f1d_alloc(&p, n) ||
      vec_f1d_alloc(&q, m) || vec_f1d_alloc(&r, m) || vec_f1d_alloc(&s, n) || vec_f1d_alloc(&q_ref, m))
  {
    fprintf(stderr, "Allocation failed");
    return 1;
  }

  init_matrix(&a);
  init_vector(&s);
  init_vector(&q);
  init_vector(&p);
  init_vector(&q_ref);
  init_vector(&r);

  if (verbose)
  {
    puts("O_ref:");
    vec_f1d_dump(&q_ref);
    puts("");

    puts("O:");
    vec_f1d_dump(&q);
    puts("");
  }

  bicg(VEC2D_ARGS(&a), VEC1D_ARGS(&p), VEC1D_ARGS(&q), VEC1D_ARGS(&r), VEC1D_ARGS(&s));

  mm_refimpl(&a, &p, &q_ref, &r, &s);

  if (verbose)
  {
    puts("Result O:");
    vec_f1d_dump(&q);
    puts("");

    puts("Reference O:");
    vec_f1d_dump(&q_ref);
    puts("");
  }

  if (!vec_f1d_compare(&q, &q_ref))
  {
    fputs("Result differs from reference result\n", stderr);
    exit(1);
  }

  vec_f2d_destroy(&a);

  vec_f1d_destroy(&s);
  vec_f1d_destroy(&p);
  vec_f1d_destroy(&q);
  vec_f1d_destroy(&q_ref);
  vec_f1d_destroy(&r);

  return 0;
}
