import pprint
import math
import time
import itertools as it
import numpy as np
from scipy.optimize import linear_sum_assignment


def get_maximum_welfare(V, flag_print_matching=False):
    """
    Given a unit-demand market, compute the maximum weight matching and value.
    :param V:
    :param flag_print_matching:
    :return:
    """
    row_ind, col_ind = linear_sum_assignment(V * -1.0)
    max_welfare = V[row_ind, col_ind].sum()
    matching = {}
    for i, j in zip(row_ind, col_ind):
        matching[i] = j
    if flag_print_matching:
        print(f'\nMaximum Weight Matching:')
        for i, j in matching.items():
            print(f'{i}->{j}: {V[i][j]}')
        print(f'max_welfare = {max_welfare} \n')

    return matching, max_welfare


def num_samples_to_epsilon(V, num_samples, delta, c, excluded_pairs):
    """
    Get the number of samples to get eps guarantee
    :param V:
    :param num_samples:
    :param delta:
    :param c:
    :param excluded_pairs:
    :return:
    """
    return c * math.sqrt(math.log((2.0 * np.size(V, 0) * np.size(V, 1) - len(excluded_pairs)) / delta) / (2.0 * num_samples))


def epsilon_to_num_samples(num_consumers, num_goods, epsilon, delta, c, excluded_pairs):
    """
    Get the epsilon guarantee given the number of samples
    :param num_consumers:
    :param num_goods
    :param epsilon:
    :param delta:
    :param c:
    :param excluded_pairs:
    :return:
    """
    return math.ceil(math.log((2.0 * num_consumers * num_goods - len(excluded_pairs)) / delta) * 0.5 * (c / epsilon) * (c / epsilon))


def elicitation_algorithm(V,
                          num_samples,
                          delta,
                          values_high,
                          values_low,
                          noise_factor,
                          excluded_pairs,
                          flag_print_debug=False):
    """
    The Elicitation Algorithm
    :param V: the unit demand market's valuation matrix
    :param num_samples: number of samples
    :param delta: failure probability
    :param values_high: the maximum number a value can take
    :param values_low: the minimum number a value can take
    :param noise_factor: defines the range of uniform noise (-noise_factor / 2, noise_factor / 2)
    :param excluded_pairs: pairs to omit for sampling
    :param flag_print_debug:
    :return:
    """
    # Make sure the noise factor is positive
    assert noise_factor >= 0.0
    if flag_print_debug:
        t0 = time.time()
        print(f'\tComputing Empirical Market ... for market \n {V}')
    # Compute the empirical market by adding noise to the market and taking the average
    empirical_market = sum([V + ((np.random.rand(np.size(V, 0), np.size(V, 1)) * noise_factor) - (noise_factor / 2.0))
                            for _ in range(0, num_samples)]) / num_samples
    for i, j in excluded_pairs:
        empirical_market[i][j] = 0.0
    if flag_print_debug:
        print(f'\tDone computing empirical market, took {time.time() - t0} sec')
    c = values_high - values_low + noise_factor
    epsilon = num_samples_to_epsilon(V, num_samples, delta, c, excluded_pairs)
    # debug prints
    if flag_print_debug:
        print('\n Call to the Elicitation Algorithm')
        print(f'empirical market =\n {empirical_market}')
        print(f'epsilon = {epsilon}')
    return empirical_market, epsilon


def can_be_pruned(V, full_empirical_welfare, consumer, good, epsilon):
    """
    Checks if the given (consumer, bundle) pair can be pruned on the given unit demand market V for error tolerance epsilon.
    :param V: a unit demand market
    :param full_empirical_welfare: the value of the estimated welfare with all consumers and goods
    :param consumer: consumer index
    :param good: good index
    :param epsilon: error tolerance
    :return: True if the pair (consumer index, good index) can be pruned, False otherwise.
    """
    # The consumer's valuation for the good
    v_ij = V[consumer][good]
    # Delete the ith row and the jth column
    sub_market = np.delete(np.delete(V, consumer, 0), good, 1)
    _, sub_market_max_welfare = get_maximum_welfare(sub_market)
    candidate_welfare = v_ij + sub_market_max_welfare + 2.0 * epsilon * np.size(V, 0)
    # Print for debug
    # print(f'\n\n sub_market = \n{sub_market}')
    # print(f'v_ij = {v_ij}, sub_market_max_welfare = {sub_market_max_welfare}')
    # print(f'cand. wel. = {candidate_welfare} ? full_emp_wel. = {full_empirical_welfare}')
    return candidate_welfare < full_empirical_welfare


def elicitation_with_pruning(V,
                             sampling_schedule,
                             delta_schedule,
                             values_high,
                             values_low,
                             noise_factor,
                             target_epsilon,
                             flag_print_debug=True):
    """
    Elicitation Algorithm with pruning.
    :param V:
    :param sampling_schedule:
    :param delta_schedule:
    :param values_high:
    :param values_low:
    :param noise_factor:
    :param target_epsilon:
    :param flag_print_debug:
    :return:
    """

    if flag_print_debug:
        print('\n\n----- START OF EAP ----')

    # We keep track of pairs (i, j) that are active, i.e., not pruned, and those that are pruned.
    active_set = set((i, j) for i, j in it.product(range(0, np.size(V, 0)), range(0, np.size(V, 1))))
    prune_set = set()

    # We keep track of the estimated values and their corresponding epsilons.
    estimated_values = {i: {j: 0 for j in range(0, np.size(V, 1))} for i in range(0, np.size(V, 0))}
    estimated_epsilons = {i: {j: math.inf for j in range(0, np.size(V, 1))} for i in range(0, np.size(V, 0))}

    # We keep track of the total number of samples used
    total_num_samples = 0

    # We keep track of pruning at each step
    total_pruned = []

    if flag_print_debug:
        print('Set Up Ready')

    # Loop through the sampling schedule and sampling deltas.
    for t, (curr_num_samples, cur_delta) in enumerate(zip(sampling_schedule, delta_schedule)):
        if flag_print_debug:
            print('Eliciting')
        cur_empirical_market, cur_epsilon = elicitation_algorithm(V=V,
                                                                  num_samples=curr_num_samples,
                                                                  delta=cur_delta,
                                                                  values_high=values_high,
                                                                  values_low=values_low,
                                                                  noise_factor=noise_factor,
                                                                  excluded_pairs=prune_set,
                                                                  flag_print_debug=flag_print_debug)
        if flag_print_debug:
            print('Done Eliciting')
        total_num_samples += len(active_set) * curr_num_samples
        # For all active pairs (i, j) update the estimated values with the most recent call to the elicitation algorithm.
        for (i, j) in active_set:
            estimated_values[i][j] = cur_empirical_market[i][j]
            estimated_epsilons[i][j] = cur_epsilon

        # Debug prints.
        if flag_print_debug:
            print(f'\n ****** iteration {t} ****** ')
            # print('Estimated Values:')
            # pprint.pprint(estimated_values)
            # print('Estimated Epss:')
            # pprint.pprint(estimated_epsilons)

        # Check the termination conditions.
        if cur_epsilon <= target_epsilon or t == len(sampling_schedule) - 1 or len(prune_set) == np.size(V, 0) * np.size(V, 1):
            if flag_print_debug:
                print(f'----- END OF EAP ---- {cur_epsilon <= target_epsilon}, {t == len(sampling_schedule) - 1}, {len(prune_set) == np.size(V, 0) * np.size(V, 1)}\n\n')
            return estimated_values, estimated_epsilons, sum([delta_schedule[l] for l in range(0, t + 1)]), cur_epsilon, total_num_samples, total_pruned

        # The termination conditions did not hold. We now try to prune. First, compute the estimated maximum welfare for our current estimated market.
        _, full_empirical_welfare = get_maximum_welfare(V=cur_empirical_market,
                                                        flag_print_matching=flag_print_debug)
        # Check pairs (i, j) that can be pruned but were not already pruned.
        cur_total_pruned = 0
        for i, j in it.product(range(0, np.size(V, 0)), range(0, np.size(V, 1))):
            if (i, j) not in prune_set:
                if flag_print_debug:
                    t0 = time.time()
                    print(f'Try pruning pair: {i},{j}: ', end='\n')
                if can_be_pruned(V=cur_empirical_market,
                                 full_empirical_welfare=full_empirical_welfare,
                                 consumer=i,
                                 good=j,
                                 epsilon=cur_epsilon):
                    prune_set.add((i, j))
                    active_set.remove((i, j))
                    cur_total_pruned += 1
                    if flag_print_debug:
                        print(f'++CAN BE PRUNED, took {time.time() - t0}')
                elif flag_print_debug:
                    print(f'--CANNOT BE PRUNED, took {time.time() - t0}')
        total_pruned += [cur_total_pruned]
        # Sanity check: the sum of the number of pruned indices and active indices must equal the size of the market.
        assert len(prune_set) + len(active_set) == np.size(V, 0) * np.size(V, 1)

        # Debug prints.
        if flag_print_debug:
            print(f'\nprune set = ')
            pprint.pprint(prune_set)
            print(f'prune set size = {len(prune_set)}')
            print(f'active set = ')
            pprint.pprint(active_set)
            print(f'active set size = {len(active_set)}')
