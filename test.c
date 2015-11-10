#include "gsl.h"

int main(int argc, char const *argv[])
{
	size_t i, n = 10;
	vector * v = &(vector){0,0,0};
	vector_alloc(v,n);
	printf("%p\n", v);
	printf("%p\n", v->data);
	for (i = 0; i < n; ++i)
		printf("%i\n", (int) v->data[i]);
	vector_free(v);

	return 0;
}