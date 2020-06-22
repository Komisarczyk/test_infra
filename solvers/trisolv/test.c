  float* l;
  float* k;
void mm_refimpl(struct vec_f2d *a, struct vec_f1d *x, struct vec_f1d *b)
{
#pragma scop
  //for (int i = 0; i < x->sizes[0]; i++)
  for (int i = 0; i < 5; i++)
  {
    l[i] = k[i];
/*    vec_f1d_set(x, i, vec_f1d_get(b, i));
    for (int j = 0; j < i; j++)
      vec_f1d_set(x, i, vec_f1d_get(x, i) - vec_f2d_get(a, i, j) * vec_f1d_get(x, j));
    vec_f1d_set(x, i, vec_f1d_get(x, i) / vec_f2d_get(a, i, i));*/
  }
#pragma endscop
}
