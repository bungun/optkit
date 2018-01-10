from optkit.compat import *

import os

from optkit import api

def establish_backend():
    if os.getenv('OPTKIT_PYTEST_GPU', False):
        api.set_backend(gpu=True)