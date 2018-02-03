from optkit.compat import *

import gc
import ctypes as ct
import itertools
import six

# from optkit.libs.linsys import DenseLinsysLibs, SparseLinsysLibs
# from optkit.libs.prox import ProxLibs
from optkit.libs.pogs import PogsDenseLibs, PogsAbstractLibs
from optkit.libs.clustering import ClusteringLibs
from optkit.utils.pyutils import version_string


# CPU32, CPU64, GPU32, GPU64
class OKBackend(object):
    __LIBNAMES = ['pogs', 'pogs_abstract', 'cluster']

    def __init__(self, gpu=False, single_precision=False):
        self.__version = None
        self.__device = None
        self.__precision = None
        self.__config = None

        self.__libs = dict()
        self.__loaders = {
                'pogs': PogsDenseLibs,
                'pogs_abstract': PogsAbstractLibs,
                'cluster': ClusteringLibs,
        }
        self.accessible_libs = [name for name in self.__LIBNAMES]
        if os.getenv('OPTKIT_NO_LOAD_ABSTRACT_POGS', False):
            self.accessible_libs.remove('pogs_abstract')
        if os.getenv('OPTKIT_NO_LOAD_CLUSTERING', False):
            self.accessible_libs.remove('cluster')

        for key in self.__loaders:
            if key in self.accessible_libs:
                self.__loaders[key] = self.__loaders[key]()
            else:
                self.__loaders[key] = None

        # # library loaders
        # # self.dense_lib_loader = DenseLinsysLibs()
        # # self.sparse_lib_loader = SparseLinsysLibs()
        # # self.prox_lib_loader = ProxLibs()
        # self.pogs_lib_loader =
        # self.pogs_abstract_lib_loader = PogsAbstractLibs()
        # self.cluster_lib_loader = ClusteringLibs()

        # # library instances
        # # self.dense = None
        # # self.sparse = None
        # # self.prox = None
        # self.pogs = None
        # self.pogs_abstract = None
        # self.cluster = None

        self.__set_lib()

    def __clear(self):
        self.__version = None
        self.__device = None
        self.__precision = None
        self.__config = '(No libraries selected)'

        self.__libs = dict()

        # # library instances
        # # self.dense = None
        # # self.sparse = None
        # # self.prox = None
        # self.pogs = None
        # self.pogs_abstract = None
        # self.cluster = None

        self.__LIBGUARD_ON = False
        self.__COBJECT_COUNT = 0

    @property
    def version(self):
        return self.__version

    @property
    def config(self):
        return self.__config

    @property
    def precision(self):
        return self.__precision

    @property
    def precision_is_32bit(self):
        return self.precision == '32'

    # redundant with above property since only 2 precision levels exist;
    # however, convenient given backend.change(...) takes argument
    # ``double`` (:obj:`bool`)
    @property
    def precision_is_64bit(self):
        return self.precision == '64'

    @property
    def device(self):
        return self.__device

    @property
    def device_is_gpu(self):
        return self.__device == 'gpu'

    @property
    def device_reset_allowed(self):
        return self.__COBJECT_COUNT == 0

    @property
    def libguard_active(self):
        return self.__LIBGUARD_ON

    def increment_cobject_count(self):
        self.__COBJECT_COUNT += 1
        self.__LIBGUARD_ON = True

    def decrement_cobject_count(self):
        self.__COBJECT_COUNT -= 1
        self.__LIBGUARD_ON = self.__COBJECT_COUNT > 0

    def __get_version(self):
        major = ct.c_int()
        minor = ct.c_int()
        change = ct.c_int()
        status = ct.c_int()
        try:
            self.pogs.optkit_version(
                    ct.byref(major), ct.byref(minor), ct.byref(change),
                    ct.byref(status))

            self.__version = 'Optkit v{}'.format(
                    version_string(major.value, minor.value, change.value,
                                   status.value))
        except:
            self.__version = 'Optkit: version unknown'

    def load_lib(self, name, override=False):
        if name not in self.__LIBNAMES:
            raise ValueError('invalid library name')
        if name in self.__libs and not override:
            print('\nlibrary {} already loaded; call with keyword arg '
                  '"override"=True to bypass this check\n'.format(name))
        self.__libs[name] = self.__loaders[name].get(
                    single_precision=self.precision_is_32bit,
                    gpu=self.device_is_gpu)

    def load_libs(self, *names):
        map(self.load_lib, names)

    def __set_lib(self, device=None, precision=None, order=None):
        self.__clear()

        devices = ['gpu', 'cpu'] if device == 'gpu' else ['cpu', 'gpu']
        precisions = ['32', '64'] if precision == '32' else ['64', '32']
        configs = itertools.product(devices, precisions)

        try:
            assert len(self.accessible_libs) > 0
            test_loader = self.__loaders[self.accessible_libs[0]]
            valid = False
            for dev, prec in configs:
                lib_key = '{}{}'.format(dev, prec)
                if test_loader.libs[lib_key] is not None:
                    valid = True
                    self.__device = dev
                    self.__precision = prec
                    self.__config = lib_key
                    map(
                            lambda nm: self.load_lib(nm, override=True),
                            self.accessible_libs)
                    break
                else:
                    print (
                            'Libraries for configuration {} '
                            'not found. Trying next configuration.'
                            ''.format(lib_key))
            assert valid
        except:
            raise RuntimeError('No libraries found for any backend')


    def change(self, gpu=False, double=True):
        if self.__LIBGUARD_ON:
            print('Backend cannot be changed once C objects have been '
                  'created.\n')
            return

        precision = '64' if double else '32'
        device = 'gpu' if gpu else 'cpu'

        self.__set_lib(device=device, precision=precision)
        self.__get_version()

    def reset_device(self):
        err_msg =    ''
        def can_reset(lib):
                return isinstance(lib, ct.CDLL) and hasattr(lib, 'ok_device_reset')
        reset_libs = list(six.moves.filter(can_reset, self.__libs.values()))

        if not self.device_reset_allowed:
            err_msg = 'device reset not allowed: C objects allocated'
        elif len(reset_libs) == 0:
            err_msg = 'device reset not possible: no libs with reset call loaded'
        else:
            if reset_libs[0].ok_reset_device() > 0:
                err_msg = 'device reset failed'

        if len(err_msg) > 0:
            raise RuntimeError(err_msg)

def add_lib(factory, libname):
    def getlib(backend):
        libs = backend._OKBackend__libs
        return libs[libname] if libname in libs else None
    setattr(factory, libname, property(getlib))
map(lambda lib: add_lib(OKBackend, lib), OKBackend._OKBackend__LIBNAMES)

