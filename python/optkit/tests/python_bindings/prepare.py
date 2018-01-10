from optkit.compat import *

import os

from optkit import api

def establish_backend():
    if int(os.getenv('OPTKIT_PYTEST_GPU', 0)):
        api.set_backend(gpu=True)