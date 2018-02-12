from optkit.compat import *

import os
import sys
import subprocess
import ctypes as ct

from optkit.libs.enums import OKEnums

def verbose_load(lib_name, lib_path, force_verbose=False):
    verbose = int(os.getenv('OPTKIT_VERBOSE_LOAD', force_verbose))
    if verbose:
        print('loading lib: {} at {}'.format(lib_name, lib_path))

def get_optkit_libdir():
    p = os.path.dirname(str(subprocess.check_output(['which', 'python'])))

    if p[:2] == 'b\'':
        p = p[2:]

    p = os.path.abspath(os.path.join(p, '..', 'lib'))
    py_version = 'python{}.{}'.format(
            sys.version_info.major, sys.version_info.minor)
    p = os.path.join(p, py_version)

    if  os.path.exists(os.path.join(p, 'dist-packages')):
        p = os.path.join(p, 'dist-packages')
    elif os.path.exists(os.path.join(p, 'site-packages')):
        p = os.path.join(p, 'site-packages')
    else:
        raise ImportError(
                'cannot locate site-packages/dist-packages to import '
                'optkit C libraries')

    return os.path.join(p, '_optkit_libs')

def patch_GOMP_cusolver(lib_path):
    """ fix for loading libcusolver.so

    error:
    "OSError: /usr/local/cuda/lib64/libcusolver.so.8.0: undefined symbol: GOMP_critical_end"

    solution:
    https://github.com/lebedov/scikit-cuda/issues/171
    """
    problem_lib = 'libcusolver.so.8.0'
    if 'linux' in sys.platform:
        if problem_lib in str(subprocess.check_output(['ldd', lib_path])):
            try:
                ct.CDLL('libgomp.so.1', mode=ct.RTLD_GLOBAL)
            except:
                pass

def retrieve_libs(lib_prefix):
    libs = {}
    global_c_build = get_optkit_libdir()
    local_c_build = os.path.abspath(os.path.join(os.path.dirname(__file__),
        '..', '..', '..', 'build'))
    search_results = '\n'
    use_local = int(os.getenv('OPTKIT_USE_LOCALLIBS', 0))

    # NB: no windows support
    ext = "dylib" if os.uname()[0] == "Darwin" else "so"

    for device in ['gpu', 'cpu']:
        for precision in ['32', '64']:
            lib_tag = '{}{}'.format(device, precision)
            lib_name = '{}{}{}.{}'.format(lib_prefix, device, precision, ext)
            lib_path = os.path.join(global_c_build, lib_name)
            if use_local or not os.path.exists(lib_path):
                lib_path = os.path.join(local_c_build, lib_name)

            if os.path.exists(lib_path):
                if device == 'gpu':
                    patch_GOMP_cusolver(lib_path)
                verbose_load(lib_name, lib_path)
                libs[lib_tag] = ct.CDLL(lib_path)
                libs[lib_tag].INITIALIZED = False
            else:
                msg = 'library {} not found at {}.\n'.format(
                        lib_name, lib_path)
                search_results += msg
                libs[lib_tag] = None

    return libs, search_results

class OptkitLibs(object):
    def __init__(self, lib_prefix, api_call_list):
        self.libs, search_results = retrieve_libs(lib_prefix)
        if all([self.libs[k] is None for k in self.libs]):
            raise ValueError(
                    'No backend libraries were located:\n{}'
                    ''.format(search_results))

        self.attach_calls = list(set(api_call_list))

    def get(self, single_precision=False, gpu=False):
        device = 'gpu' if gpu else 'cpu'
        precision = '32' if single_precision else '64'
        lib_key = '{}{}'.format(device, precision)
        if lib_key not in self.libs:
            return None
        elif self.libs[lib_key] is None:
            return None

        lib = self.libs[lib_key]
        if lib.INITIALIZED:
            return lib
        else:
            lib.enums = OKEnums()

            for attach_call in self.attach_calls:
                attach_call(lib, single_precision=single_precision)

            lib.FLOAT = single_precision
            lib.GPU = gpu
            lib.INITIALIZED = True
            return lib

    @staticmethod
    def conditional_include(lib, token, include_call, **include_args):
        if not token in dir(lib):
            include_call(lib, **include_args)

    @staticmethod
    def attach_default_restype(*methods):
        for m in methods:
            m.restype = ct.c_uint