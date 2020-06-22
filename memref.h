#ifndef MEMREF_H
#define MEMREF_H

#include <stdint.h>
#include <stdlib.h>

/* Data layout information for a 1d float memref */
struct vec_f1d
{
	float *allocatedPtr;
	float *alignedPtr;
	int64_t offset;
	int64_t sizes[1];
	int64_t strides[1];
};

/* Data layout information for a 2d float memref */
struct vec_f2d
{
	float *allocatedPtr;
	float *alignedPtr;
	int64_t offset;
	int64_t sizes[2];
	int64_t strides[2];
};
/* Data layout information for a 3d float memref */
struct vec_f3d
{
	float *allocatedPtr;
	float *alignedPtr;
	int64_t offset;
	int64_t sizes[3];
	int64_t strides[3];
};

/* Generates a comma-separated list of arguments from the fields of a
 * 1d float memref */
#define VEC1D_ARGS(v)    \
	(v)->allocatedPtr,   \
		(v)->alignedPtr, \
		(v)->offset,     \
		(v)->sizes[0],   \
		(v)->strides[0]

/* Generates a comma-separated list of arguments from the fields of a
 * 2d float memref */
#define VEC2D_ARGS(v)    \
	(v)->allocatedPtr,   \
		(v)->alignedPtr, \
		(v)->offset,     \
		(v)->sizes[0],   \
		(v)->sizes[1],   \
		(v)->strides[0], \
		(v)->strides[1]

/* Generates a comma-separated list of arguments from the fields of a
 * 2d float memref */
#define VEC3D_ARGS(v)    \
	(v)->allocatedPtr,   \
		(v)->alignedPtr, \
		(v)->offset,     \
		(v)->sizes[0],   \
		(v)->sizes[1],   \
		(v)->sizes[2],   \
		(v)->strides[0], \
		(v)->strides[1], \
		(v)->strides[2]
/* Allocates and initializes a 1d float memref. Returns 0 on success,
 * otherwise 1.
 */
static inline int vec_f1d_alloc(struct vec_f1d *v, size_t n)
{
	float *f;

	if (!(f = calloc(n, sizeof(float))))
		return 1;

	v->allocatedPtr = f;
	v->alignedPtr = f;
	v->offset = 0;
	v->sizes[0] = n;
	v->strides[0] = n;

	return 0;
}

/* Allocates and initializes a 2d float memref. Returns 0 on success,
 * otherwise 1.
 */
static inline int vec_f2d_alloc(struct vec_f2d *v, size_t n, size_t m)
{
	float *f;

	if (!(f = calloc(n * m, sizeof(float))))
		return 1;

	v->allocatedPtr = f;
	v->alignedPtr = f;
	v->offset = 0;
	v->sizes[0] = n;
	v->sizes[1] = m;
	v->strides[0] = 1;
	v->strides[1] = m;

	return 0;
}
/* Allocates and initializes a 3d float memref. Returns 0 on success,
 * otherwise 1.
 */
static inline int vec_f3d_alloc(struct vec_f3d *v, size_t m, size_t n, size_t o)
{
	float *f;

	if (!(f = calloc(n * m * o, sizeof(float))))
		return 1;

	v->allocatedPtr = f;
	v->alignedPtr = f;
	v->offset = 0;
	v->sizes[0] = m;
	v->sizes[1] = n;
	v->sizes[2] = o;
	v->strides[0] = 1;
	v->strides[1] = 1;
	v->strides[2] = o;

	return 0;
}
/* Destroys a 1d float memref */
static inline void vec_f1d_destroy(struct vec_f1d *v)
{
	free(v->allocatedPtr);
}

/* Destroys a 2d float memref */
static inline void vec_f2d_destroy(struct vec_f2d *v)
{
	free(v->allocatedPtr);
}

/* Destroys a 3d float memref */
static inline void vec_f3d_destroy(struct vec_f3d *v)
{
	free(v->allocatedPtr);
}
/* Returns the element at position (`x`) of a 1d float memref `v` */
static inline float vec_f1d_get(struct vec_f1d *v, int64_t x)
{
	return *(v->allocatedPtr + x);
}

/* Returns the element at position (`x`, `y`) of a 2d float memref `v` */
static inline float vec_f2d_get(struct vec_f2d *v, int64_t x, int64_t y)
{
	return *(v->allocatedPtr + x * v->sizes[1] + y);
}

/* Returns the element at position (`x`, `y`,'z') of a 3d float memref `v` */
static inline float vec_f3d_get(struct vec_f3d *v, int64_t x, int64_t y, int64_t z)
{
	return *(v->allocatedPtr + x * v->sizes[1] * v->sizes[2] + y * v->sizes[2] + z);
}

/* Assigns `f` to the element at position (`x`) of a 1d float
 * memref `v`
 */
static inline void vec_f1d_set(struct vec_f1d *v, int64_t x, float f)
{
	*(v->allocatedPtr + x) = f;
}

/* Assigns `f` to the element at position (`x`, `y`) of a 2d float
 * memref `v`
 */
static inline void vec_f2d_set(struct vec_f2d *v, int64_t x, int64_t y, float f)
{
	*(v->allocatedPtr + x * v->sizes[1] + y) = f;
}

/* Assigns `f` to the element at position (`x`, `y`,'z') of a 3d float
 * memref `v`
 */
static inline void vec_f3d_set(struct vec_f3d *v, int64_t x, int64_t y, int64_t z, float f)
{
	*(v->allocatedPtr + x * v->sizes[1] * v->sizes[2] + y * v->sizes[2] + z) = f;
}
/* Compares the values of two 1d float memrefs. Returns 1 if they are
 * equal, otherwise 0.
 */
static inline int vec_f1d_compare(struct vec_f1d *a, struct vec_f1d *b)
{
	/* Compare shapes */
	if (a->sizes[0] != b->sizes[0])
	{
		return 0;
	}

	/* Compare elements */
	for (int64_t x = 0; x < a->sizes[0]; x++)
		if (vec_f1d_get(a, x) != vec_f1d_get(b, x))
			return 0;

	return 1;
}

/* Compares the values of two 2d float memrefs. Returns 1 if they are
 * equal, otherwise 0.
 */
static inline int vec_f2d_compare(struct vec_f2d *a, struct vec_f2d *b)
{
	/* Compare shapes */
	if (a->sizes[0] != b->sizes[0] ||
		a->sizes[1] != b->sizes[1])
	{
		return 0;
	}

	/* Compare elements */
	for (int64_t y = 0; y < a->sizes[1]; y++)
		for (int64_t x = 0; x < a->sizes[0]; x++)
			if (vec_f2d_get(a, x, y) != vec_f2d_get(b, x, y)){
				printf("%f    %f \n",vec_f2d_get(a, x, y),vec_f2d_get(b, x, y));
				return 0;
			}

	return 1;
}
/* Compares the values of two 3d float memrefs. Returns 1 if they are
 * equal, otherwise 0.
 */
static inline int vec_f3d_compare(struct vec_f3d *a, struct vec_f3d *b)
{
	/* Compare shapes */
	if (a->sizes[0] != b->sizes[0] ||
		a->sizes[1] != b->sizes[1] ||
		a->sizes[2] != b->sizes[2])
	{
		return 0;
	}

	/* Compare elements */
	for (int64_t z = 0; z < a->sizes[2]; z++)
		for (int64_t y = 0; y < a->sizes[1]; y++)
			for (int64_t x = 0; x < a->sizes[0]; x++)
				if (vec_f3d_get(a, x, y, z) != vec_f3d_get(b, x, y, z))
					return 0;

	return 1;
}
/* Dumps a 1d float memref `v` to stdout. */
static inline void vec_f1d_dump(struct vec_f1d *v)
{

	for (int64_t x = 0; x < v->sizes[0]; x++)
	{
		printf("%f%s",
			   *(v->allocatedPtr + x), " ");
	}

	puts("");
}

/* Dumps a 2d float memref `v` to stdout. */
static inline void vec_f2d_dump(struct vec_f2d *v)
{

	for (int64_t x = 0; x < v->sizes[0]; x++)
	{
		for (int64_t y = 0; y < v->sizes[1]; y++)
		{
			printf("%f%s",
				   *(v->allocatedPtr + x * v->sizes[1] + y),
				   y == v->sizes[1] - 1 ? "" : " ");
		}

		puts("");
	}
}

/* Dumps a 3d float memref `v` to stdout. */
static inline void vec_f3d_dump(struct vec_f3d *v)
{

	for (int64_t x = 0; x < v->sizes[0]; x++)
	{
		for (int64_t y = 0; y < v->sizes[1]; y++)
		{
			for (int64_t z = 0; z < v->sizes[2]; z++)
			{
				printf("%f%s",
					   *(v->allocatedPtr + x * v->sizes[1] * v->sizes[2] + y * v->sizes[2] + z),
					   z == v->sizes[2] - 1 ? "" : " ");
			}
		}
		puts("");
	}
}
#endif
