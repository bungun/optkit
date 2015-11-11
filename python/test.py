from optkit import *

# Script
# ------

v = make_cvector()
w = make_cvector()

oklib.__vector_calloc(v, 10)
oklib.__vector_calloc(w, 10)
oklib.__vector_add_constant(v, 12.)
oklib.__vector_add_constant(w, 5.)
oklib.__blas_axpy(1., v , w)
oklib.__vector_print(v)
oklib.__vector_print(w)
oklib.__vector_free(v)
oklib.__vector_free(w)



