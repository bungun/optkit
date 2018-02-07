import optkit as ok

from pogs_basis_pursuit import basis_pursuit
from pogs_entropy_maximization import entropy_maximization
from pogs_huber_fitting import huber_fitting
from pogs_lasso import lasso
from pogs_linear_program import linear_program
from pogs_logistic_regression import logistic_regression
from pogs_nnls import nonnegative_least_squares
from pogs_portfolio import portfolio_optimization
from pogs_svm import support_vector_machine
from pogs_radiation import intensity_optimization

PROBLEM_CLASSES = dict(
        basis_pursuit=basis_pursuit,
        entropy_maximization=entropy_maximization,
        huber_fitting=huber_fitting,
        lasso=lasso,
        linear_program=linear_program,
        logistic_regression=logistic_regression,
        nonnegative_least_squares=nonnegative_least_squares,
        portfolio_optimization=portfolio_optimization,
        support_vector_machine=support_vector_machine,
        intensity_optimization=intensity_optimization)

def run_once(shape, problem_class, **options):
    A, f, g = eval(problem_class)(shape, **options)
    with ok.api.PogsSolver(A) as solver:
        solver.solve(f, g, **options)
        return solver.info
