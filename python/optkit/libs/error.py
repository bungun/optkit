from optkit.compat import *

import collections

error_codes = {
    0: 'OPTKIT SUCCESS'
    1: 'OPTKIT ERROR'
    2: 'OPTKIT ERROR CUDA'
    3: 'OPTKIT ERROR CUBLAS'
    4: 'OPTKIT ERROR CUSPARSE'
    5: 'OPTKIT ERROR LAPACK'
    10: 'OPTKIT ERROR DOMAIN'
    11: 'OPTKIT ERROR DIVIDE BY ZERO'
    100: 'OPTKIT ERROR LAYOUT MISMATCH'
    101: 'OPTKIT ERROR DIMENSION MISMATCH'
    102: 'OPTKIT ERROR OUT OF BOUNDS'
    1000: 'OPTKIT ERROR OVERWRITE'
    1001: 'OPTKIT ERROR UNALLCOATED'
    2000: 'OPTKIT ERROR NOT IMPLEMENTED'
}

def optkit_print_error(err, print_success=False):
    if err is None:
        return
    elif err == 0:
        if print_success:
            print('\nOPTKIT SUCCESS')
    elif err in error_codes:
        print('\n{}'.format(error_codes[err]))
    else:
        print('\nunrecognized error code: {}'.format(err))
    return err