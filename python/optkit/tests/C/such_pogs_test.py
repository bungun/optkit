with such.A('POGS Dense Library') as it:
    @it.should('retrieve one or more C POGS libraries')

    @it.should('init and finish')
    @params()
    def test(case, order):
        pass

    with it.having('a solver object'):
        @it.has_setup
        def setup():
            pass

        @it.has_teardown
        def teardown():
            pass

        @it.has_test_setup
        def test_setup(case):
            pass

        @it.has_test_teardown
        def test_teardown(case):
            pass

        @it.should('initialize conditions')
        @it.should('equilibrate problem matrix as A_equil = DAE')
        @it.should('apply matrix')
        @it.should('apply adjoint')
        @it.should('project onto y = DAEx')
        @it.should('scale input functions')
        @it.should('update primal variables')
        @it.should('prox primal variables')
        @it.should('project primal variables')
        @it.should('update dual variables')
        @it.should('check for convergence')
        @it.should('adapt rho')
        @it.should('unscale outputs')
        @it.should('solve end-to-end')
        @it.should('produce converged unscaled outputs')
        @it.should('scale warmstart variables')
        @it.should('warmstart with shorter solves than coldstart')
        @it.should('cache and retrieve state')

    @it.should('solve and complete without passing user a solver pointer')


