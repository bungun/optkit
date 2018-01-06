from optkit.compat import *

import optkit.libs.error as okerr
from optkit.tests.statements import *

def noerr(c_call_status):
    return okerr.optkit_print_error(c_call_status) == 0