import itertools as it
import pulp
import numpy as np


def solve_ce_prices_lp(V, matching, minimize=True, flag_debug_print=False):
    """
    Given a unit-demand market and a matching, solve for CE prices.
    :param V:
    :param matching:
    :param minimize:
    :param flag_debug_print:
    :return:
    """
    # Construct the prices LP
    ce_prices_lp = pulp.LpProblem('CEPrices', pulp.LpMinimize if minimize else pulp.LpMaximize)
    # Generate variables
    prices_vars = pulp.LpVariable.dicts("price",
                                        [j for j in range(0, np.size(V, 1))],
                                        lowBound=0.0,
                                        cat='Continuous')
    # Generate objective
    ce_prices_lp += pulp.lpSum([prices_vars[j] for j in range(0, np.size(V, 1))])

    # Prices bounded below by 0
    for j in range(0, np.size(V, 1)):
        ce_prices_lp += prices_vars[j] >= 0.0

    # Generate market clearance constraints
    allocated_items = matching.values()
    for j in range(0, np.size(V, 1)):
        if j not in allocated_items:
            ce_prices_lp += prices_vars[j] == 0.0

    # Generate utility maximization constraints. These do not include I.R., which we do next.
    for i, j in it.product(range(0, np.size(V, 0)), range(0, np.size(V, 1))):
        if i not in matching:
            ce_prices_lp += 0 >= V[i][j] - prices_vars[j]
        elif j != matching[i]:
            ce_prices_lp += V[i][matching[i]] - prices_vars[matching[i]] >= V[i][j] - prices_vars[j]

    # Generate I.R. constraints.
    for i in range(0, np.size(V, 0)):
        if i in matching:
            ce_prices_lp += V[i][matching[i]] - prices_vars[matching[i]] >= 0

    # Solve LP and find solution
    ce_prices_lp.solve()
    final_prices = [prices_vars[j].varValue for j in range(0, np.size(V, 1))]
    if flag_debug_print:
        print(f'\n\nminimize = {minimize}, status = {pulp.LpStatus[ce_prices_lp.status]}')
        # print(ce_prices_lp)
        print(f'final_prices = {final_prices} \n\n')
    return final_prices


def compute_market_regret(V, matching, prices, flag_debug_print=False):
    """
    Computes the maximum market regret for the given outcome (matching, prices).
    The maximum market regret is defined as the maximum regret over all consumers.
    The maximum regret of a consumer is the best utility it could have gotten.
    :param V:
    :param matching:
    :param prices:
    :param flag_debug_print:
    :return:
    """
    max_consumers_utility = [max([V[i][j] - prices[j] for j in range(0, np.size(V, 1))]) for i in range(0, np.size(V, 0))]
    max_market_regret = max([max_cons_regert - (V[i][matching[i]] - prices[matching[i]] if i in matching else 0.0) for i, max_cons_regert in enumerate(max_consumers_utility)])
    if flag_debug_print:
        print(f'max_consumers_utility = {max_consumers_utility}')
        print(f'max_market_regret = {max_market_regret}')
    return max_market_regret
