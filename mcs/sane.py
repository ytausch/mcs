from dataclasses import dataclass
from typing import Callable

import numpy as np

from mcs import mcs
from mcs.functions import functions


@dataclass
class MCSResult:
    f_best: float
    x_best: np.ndarray
    function_call_count: int


def run_mcs(
        f: Callable[[np.ndarray], float],
        domain_lower: list[float],
        domain_upper: list[float],
        number_of_levels: int | None = None,
        function_call_limit: int | None = None,
        stopping_criteria: list[float] | None = None,
        initialization_type: int = 0,
        local_search: int = 50,
        local_search_accuracy: float = 2.220446049250313e-16,
        hessian_sparsity: np.ndarray = None
) -> MCSResult:
    """
    Run the MCS algorithm to find the global minimum of a function.
    :param f: The function to minimize.
    :param domain_lower: The lower bounds of the domain.
    :param domain_upper: The upper bounds of the domain.
    :param number_of_levels: The number of levels used.
    :param function_call_limit: The limit on the number of function calls.
    :param stopping_criteria: The stopping criteria. The meaning of the values is currently undefined.
    Defaults to [3 * n, float("-inf")] where n is the number of dimensions.
    :param initialization_type: The initialization type. Defaults to 0. 0: simple initialization list.
    :param local_search: The local search parameter. Defaults to 50. 0: no local search.
    :param local_search_accuracy: The acceptable relative accuracy for local search.
    :param hessian_sparsity: The sparsity pattern of the Hessian. Defaults to a nxn matrix of ones.
    :return: The best function value and the corresponding input.
    """
    def target_function(x: np.ndarray | list) -> float:
        if isinstance(x, list):
            # yep, this can happen...
            x = np.array(x)
        return f(x)

    # monkeypatching this is a very ugly hack but avoids modifying the existing code too much
    functions.myfun = target_function

    n = len(domain_lower)
    """
    number of dimensions
    """

    if len(domain_upper) != n:
        raise ValueError('domain_lower and domain_upper must have the same length')

    x_best, f_best, _, _, n_calls, _, _ = mcs.mcs(
        'myfun',
        domain_lower,
        domain_upper,
        number_of_levels or 5 * n + 10,
        function_call_limit or 50 * n ** 2,
        stopping_criteria or [3 * n, float("-inf")],
        initialization_type,
        local_search,
        local_search_accuracy,
        hessian_sparsity or np.ones((n, n))
    )

    return MCSResult(f_best, x_best, n_calls)
