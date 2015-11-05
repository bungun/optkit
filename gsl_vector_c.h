#ifndef GSL_VECTOR_H_GUARD
#define GSL_VECTOR_H_GUARD

#include <cstdio>
#include <cstring>
#include "gsl_defs.h"

#ifdef __cplusplus
extern "C" {
#endif



typedef struct vector_s {
	size_t size, stride;
	float *data;
} vector_s;

typedef struct vector_d {
	size_t size, stride;
	double *data;
} vector_d;


vector_s vector_alloc_s(size_t n) {
	return (vector_s){
		.size=n,
		.stride=1,
		.data=(float *) malloc(n * sizeof(float))
	};
}

vector_d vector_alloc_d(size_t n) {
	return (vector_d){
		.size=n,
		.stride=1,
		.data=(double *) malloc(n * sizeof(double))
	};
}

vector_s vector_calloc_s(size_t n) {
	vector_s vec = vector_alloc_s(n);
	memset(vec.data,0,n * sizeof(float));
	return vec;
}

vector_d vector_calloc_d(size_t n) {
	vector_d vec = vector_alloc_d(n);
	memset(vec.data,0,n * sizeof(double));
	return vec;
}

void vector_free_s(vector_s *x) {
	free(x->data);
}

void vector_free_d(vector_d *x) {
	free(x->data);
}

inline void vector_set_s(vector_s *v, size_t i, float x) {
	v->data[i * v->stride] = x;
}

inline void vector_set_d(vector_d *v, size_t i, double x) {
	v->data[i * v->stride] = x;
}

inline float vector_get_s(vector_s *v, size_t i) {
	return v->data[i * v->stride];
}

inline double vector_get_d(vector_d *v, size_t i) {
	return v->data[i * v->stride];
}

void vector_set_all_s(vector_s *v, float x) {
	for (unsigned int i = 0; i < v->size; ++i)
		vector_set(v, i , x);
}

void vector_set_all_d(vector_d *v, double x) {
	for (unsigned int i = 0; i < v->size; ++i)
		vector_set(v, i , x);
}

vector_s vector_subvector_s(vector_s *vec, size_t offset, size_t n) {
	return (vector_s){
		.size = n,
		.stride = vec->stride, 
		.data = vec->data + offset * vec->stride;
	};
}

vector_d vector_subvector_d(vector_d *vec, size_t offset, size_t n) {
	return (vector_d){
		.size = n,
		.stride = vec->stride, 
		.data = vec->data + offset * vec->stride
	};
}

vector_s vector_view_array_s(float *base, size_t n) {
	return (vector_s){
		.size = n,
		.stride = vec->stride,
		.data = base
	}
}

vector_d vector_view_array_d(double *base, size_t n) {
	return (vector_d){
		.size = n,
		.stride = vec->stride,
		.data = base
	}
}

void vector_memcpy_vv_s(vector_s *x, const vector_s *y) {
	if (x->stride) == 1 && y->stride == 1) {
		memcpy(x->data, y->data, x->size * sizeof(float))
	} else {
		for (unsigned int i = 0; i < count; ++i)
			vector_set_s(x, i, y[i])
	}
}

void vector_memcpy_vv_d(vector_d *x, const vector_d *y) {
	if (x->stride) == 1 && y->stride == 1) {
		memcpy(x->data, y->data, x->size * sizeof(double))
	} else {
		for (unsigned int i = 0; i < count; ++i)
			vector_set_d(x, i, y[i])
	}
}

void vector_memcpy_va_s(vector_s *x, const float *y) {
	if (x->stride == 1) {
		memcpy(x->data, y, x->size * sizeof(float));
	} else {
		for (unsigned int i = 0; i < x->size; ++i)
			vector_set_s(x, i, y[i]);
	}
}

void vector_memcpy_va_d(vector_d *x, const double *y) {
	if (x->stride == 1) {
		memcpy(x->data, y, x->size * sizeof(double));
	} else {
		for (unsigned int i = 0; i < x->size; ++i)
			vector_set_d(x, i, y[i]);
	}
}


void vector_memcpy_av_s(float *x, const vector_s *y) {
	if (y->stride ==1) {
		memcpy(x, y->data, y->size * sizeof(float));
	} else {
		for (unsigned int i = 0; i < y->size; ++i)
			x[i] = vector_get(y,i);
	}
}

void vector_memcpy_av_d(double *x, const vector_d *y) {
	if (y->stride ==1) {
		memcpy(x, y->data, y->size * sizeof(double));
	} else {
		for (unsigned int i = 0; i < y->size; ++i)
			x[i] = vector_get(y,i);
	}
}


void vector_print_s(const vector_s *x) {
	for (unsigned int i = 0; i < count; ++i)
		printf("%e ", vector_get_s(x, i));
	printf("\n");
}

void vector_print_d(const vector_d *x) {
	for (unsigned int i = 0; i < count; ++i)
		printf("%e ", vector_get_d(x, i));
	printf("\n");
}

void vector_scale_s(vector_s *a, float x) {
	for (unsigned int i = 0; i < count; ++i)
		a->data[i * a->stride] *= x;
}

void vector_scale_d(vector_d *a, double x) {
	for (unsigned int i = 0; i < count; ++i)
		a->data[i * a->stride] *= x;
}

void vector_add_s(vector_s *a, const vector_s *b) {
	for (unsigned int i = 0; i < a->size; ++i)
		a->data[i * a->stride] += b->data[i * b->stride];
}

void vector_add_s(vector_d *a, const vector_d *b) {
	for (unsigned int i = 0; i < a->size; ++i)
		a->data[i * a->stride] += b->data[i * b->stride];
}

void vector_sub_s(vector_s *a, const vector_s *b) {
	for (unsigned int i = 0; i < a->size; ++i)
		a->data[i * a->stride] -= b->data[i * b->stride];
}

void vector_sub_d(vector_d *a, const vector_d *b) {
	for (unsigned int i = 0; i < a->size; ++i)
		a->data[i * a->stride] -= b->data[i * b->stride];
}

void vector_mul_s(vector_s *a, const vector_s *b) {
	for (unsigned int i = 0; i < a->size; ++i)
		a->data[i * a->stride] *= b->data[i * b->stride];
}

void vector_mul_d(vector_d *a, const vector_d *b) {
	for (unsigned int i = 0; i < a->size; ++i)
		a->data[i * a->stride] *= b->data[i * b->stride];
}

void vector_div_s(vector_s *a, const vector_s *b) {
	for (unsigned int i = 0; i < a->size; ++i)
		a->data[i * a->stride] /= b->data[i * b->stride];
}

void vector_div_d(vector_d *a, const vector_d *b) {
	for (unsigned int i = 0; i < a->size; ++i)
		a->data[i * a->stride] /= b->data[i * b->stride];
}

void vector_add_constant_s(vector_s *a, const float x) {
	for (unsigned int i = 0; i < a->size; ++i)
		a->data[i * a->stride] += b->data[i * b->stride];
}

void vector_add_constant_d(vector_d *a, const double x) {
	for (unsigned int i = 0; i < a->size; ++i)
		a->data[i * a->stride] += x;
}


#ifdef __cplusplus
}
#endif

#endif  // GSL_VECTOR_H_GUARD

