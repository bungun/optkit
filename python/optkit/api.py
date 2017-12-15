from optkit.compat import *

import os

from optkit.backends import OKBackend
from optkit.types import PogsTypes, PogsAbstractTypes, ClusteringTypes

"""
Version query
"""
OPTKIT_VERSION = None

"""
Backend handle
"""
backend = OKBackend()

"""
C implementations
"""
pogs_types = None
PogsSolver = None
PogsObjective = None

pogs_abstract_types = None
PogsAbstractSolver = None

clustering_types = None
ClusteringSettings = None
Clustering = None

"""
Backend switching
"""
def set_backend(gpu=False, double=True):

    # Backend
    global OPTKIT_VERSION
    global backend

    ## C implementations
    global pogs_types
    global PogsSolver
    global PogsObjective

    global pogs_abstract_types
    global PogsAbstractSolver

    global clustering_types
    global ClusteringSettings
    global Clustering

    # change backend
    backend_name = backend.change(gpu=gpu, double=double)
    device = 'gpu' if gpu else 'cpu'
    precision = '64' if double else '32'
    requested_config = '{}{}'.format(device, precision)

    OPTKIT_VERSION = backend.version

    ## C implemenetations
    pogs_types = PogsTypes(backend)
    PogsSolver = pogs_types.Solver
    PogsObjective = pogs_types.Objective

    pogs_abstract_types = PogsAbstractTypes(backend)
    PogsAbstractSolver = pogs_abstract_types.Solver

    clustering_types = ClusteringTypes(backend)
    ClusteringSettings = clustering_types.ClusteringSettings
    Clustering = clustering_types.Clustering

    print('optkit backend set to {}'.format(backend.config))
    return int(requested_config != backend.config)

def clear_backend():
    global backend
    backend.reset_device()

"""
INITIALIZATION BEHAVIOR:
"""

default_device = os.getenv('OPTKIT_DEFAULT_DEVICE', 'cpu')
default_precision = os.getenv('OPTKIT_DEFAULT_FLOATBITS', '64')

set_backend(gpu=(default_device == 'gpu'),
            double=(default_precision == '64'))

del OKBackend