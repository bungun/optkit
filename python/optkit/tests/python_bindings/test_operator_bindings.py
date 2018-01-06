from optkit.compat import *

import gc
import os
import numpy as np
import scipy.sparse as sp
import unittest

from optkit.api import backend
from optkit.types.operator import *

class OperatorBindingsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def setUp(self):
        self.A_test = self.A_test_gen
        self.A_test_csr = sp.csr_matrix(self.A_test_sparse_gen)
        self.A_test_csc = sp.csc_matrix(self.A_test_sparse_gen)
        self.A_test_coo = sp.coo_matrix(self.A_test_sparse_gen)

        self.matrices = [
            self.A_test, self.A_test_csr, self.A_test_csc, self.A_test_coo,
        ]

    def test_abstract_linear_operator(self):
        lib = backend.pogs_abstract
        ALO = OperatorTypes(backend, lib).AbstractLinearOperator
        for A in self.matrices:
            O = ALO(A)
            self.assertFalse( backend.device_reset_allowed )
            del O
            self.assertTrue( backend.device_reset_allowed )