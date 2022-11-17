# do it with several solvers
import numpy as np
import six

try:
    import ortools
except ImportError:
    ortools_available = False
else:
    from ortools.linear_solver import pywraplp

    ortools_available = True


ORTOOLS = "ortools"
GUROBI = "gurobi"


def bound_A(x_min, x_max, m, W_up, b_up, W_low, b_low, mask, solver=ORTOOLS):

    func = get(solver)
    return func(x_min, x_max, m, W_up, b_up, W_low, b_low, mask)


def bound_B(x_min, x_max, m, W_up, b_up, W_low, b_low, mask, solver=ORTOOLS):

    return bound_B_ortools(x_min, x_max, m, W_low, b_low, W_up, b_up, mask)


def deserialize(name):
    """Get the activation from name.

    :param name: name of the method.
    among the implemented Keras activation function.
    :return:

    """
    name = name.lower()

    if name == ORTOOLS:
        return bound_A_ortools

    raise ValueError("Could not interpret " "function identifier:", name)


def get(identifier):
    """Get the `identifier` activation function.

    :param identifier: None or str, name of the function.
    :return: The activation function, `linear` if `identifier` is None.
    :raises: ValueError if unknown identifier

    """
    if identifier is None:
        raise ValueError()
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize(identifier)


def bound_A_ortools(x_min, x_max, m, W_up, b_up, W_low, b_low, mask_A):
    """
    x_min: (n_batch, n_dim)
    x_max: (n_batch, n_dim)
    m: discrete positive constant
    W_up : (n_batch, n_dim, n_out)
    b_up: (None, n_out)
    W_low : (None, n_dim, n_out)
    b_low: (None, n_out)
    """
    # Check ortools available
    if not ortools_available:
        raise ImportError("'ortools' must be installed to use this function.")

    # compute a mask that removes the constraints where we know it is not achievable
    n_out = W_up.shape[2]
    n_dim = W_up.shape[1]
    n_batch = W_up.shape[0]

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")
    # lambda are binary variables.

    lambda_ = [
        [[solver.IntVar(0, m, "lambda_{}_{}_{}".format(j, i, k)) for i in range(n_dim)] for k in range(n_out)]
        for j in range(n_batch)
    ]

    z = [
        [[x_min[j, i] + lambda_[j][k][i] * (x_max[j, i] - x_min[j, i]) / m for i in range(n_dim)] for k in range(n_out)]
        for j in range(n_batch)
    ]

    obj = sum(
        [
            sum([sum([W_up[j, i, k] * z[j][k][i] for i in range(n_dim)]) + b_up[j][k] for k in range(n_out)])
            for j in range(n_batch)
        ]
    )

    for j in range(n_batch):
        for k in range(n_out):
            if not mask_A[j, k]:
                const_kj = sum([W_low[j, i, k] * z[j][k][i] for i in range(n_dim)]) + b_low[j, k]
                solver.Add(const_kj <= 0)
    solver.Maximize(obj)

    status = solver.Solve()

    A = np.zeros((n_batch, n_out))
    if status == pywraplp.Solver.OPTIMAL:

        # retrieve per output and per batch samples
        for j in range(n_batch):
            for k in range(n_out):
                if not mask_A[j, k]:
                    A_jk = (
                        sum(
                            [
                                W_up[j, i, k]
                                * (x_min[j, i] + lambda_[j][k][i].solution_value() * (x_max[j, i] - x_min[j, i]) / m)
                                for i in range(n_dim)
                            ]
                        )
                        + b_up[j][k]
                    )
                    A[j, k] = A_jk
    else:
        import pdb

        pdb.set_trace()

    return A * (1 - mask_A)


def bound_B_ortools(x_min, x_max, m, W_up, b_up, W_low, b_low, mask_B):
    """
    x_min: (n_batch, n_dim)
    x_max: (n_batch, n_dim)
    m: discrete positive constant
    W_up : (n_batch, n_dim, n_out)
    b_up: (None, n_out)
    W_low : (None, n_dim, n_out)
    b_low: (None, n_out)
    """
    # Check ortools available
    if not ortools_available:
        raise ImportError("'ortools' must be installed to use this function.")

    n_out = W_up.shape[2]
    n_dim = W_up.shape[1]
    n_batch = W_up.shape[0]

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")
    # lambda are binary variables.
    lambda_ = [
        [[solver.IntVar(0, m, "lambda_{}_{}_{}".format(j, i, k)) for i in range(n_dim)] for k in range(n_out)]
        for j in range(n_batch)
    ]

    z = [
        [[x_min[j, i] + lambda_[j][k][i] * (x_max[j, i] - x_min[j, i]) / m for i in range(n_dim)] for k in range(n_out)]
        for j in range(n_batch)
    ]

    obj = sum(
        [
            sum([sum([W_up[j, i, k] * z[j][k][i] for i in range(n_dim)]) + b_up[j][k] for k in range(n_out)])
            for j in range(n_batch)
        ]
    )

    for j in range(n_batch):
        for k in range(n_out):
            if not mask_B[j, k]:
                const_kj = sum([W_low[j, i, k] * z[j][k][i] for i in range(n_dim)]) + b_low[j, k]
                solver.Add(const_kj >= 0)
    solver.Minimize(obj)

    status = solver.Solve()

    A = np.zeros((n_batch, n_out))
    if status == pywraplp.Solver.OPTIMAL:

        # retrieve per output and per batch samples
        for j in range(n_batch):
            for k in range(n_out):
                if not mask_B[j, k]:
                    A_jk = (
                        sum(
                            [
                                W_up[j, i, k]
                                * (x_min[j, i] + lambda_[j][k][i].solution_value() * (x_max[j, i] - x_min[j, i]) / m)
                                for i in range(n_dim)
                            ]
                        )
                        + b_up[j][k]
                    )
                    A[j, k] = A_jk
    else:
        import pdb

        pdb.set_trace()

    return A * (1 - mask_B)
