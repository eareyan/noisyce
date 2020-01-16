import pprint
import math
import itertools as it
import numpy as np
from scipy import sparse
import networkx as nx
from networkx.algorithms import bipartite


def print_graph(G):
    """
    Prints information about the given graph
    :param G: a nx graph object
    :return: None
    """
    pp = pprint.PrettyPrinter(indent=4)
    print(f"G = {G}")
    print("Nodes in G")
    pp.pprint(list(G.nodes(data=True)))
    print("Edges in G: ")
    pp.pprint(list(G.edges(data=True)))
    print(f"is it bipartite? = {nx.is_bipartite(G)}")


def add_noise(V, exclude_pairs, noise_scale):
    V = np.copy(V)
    for consumer, good in it.product(range(np.size(V, 0)), range(np.size(V, 1))):
        if (consumer, good) in exclude_pairs:
            V[consumer][good] = 0.0
        else:
            # TODO other noise models.
            V[consumer][good] = V[consumer][good] + np.random.rand() * noise_scale
    return V


def get_maximum_welfare(V, flag_print_matching=False):
    """
    Compute the maximum welfare of a unit demand market.
    :param V: the unit demand market
    :param flag_print_matching:
    :return: the maximum welfare
    """
    G = bipartite.from_biadjacency_matrix(sparse.coo_matrix(V))
    # print_graph(G)
    matching = nx.max_weight_matching(G)
    # We standarize indice to 0, 1, ..., num_consumers for consumers, and 0, 1, ..., num_goods for goods.
    re_index_match = {}
    for u, v in matching:
        if u < np.size(V, 0):
            re_index_match[u] = v - np.size(V, 0)
        else:
            re_index_match[v] = u - np.size(V, 0)
    matching = re_index_match
    weight = sum([V[i][j] for i, j in matching.items()])
    if flag_print_matching:
        print(f'\nMaximum Weight Matching:')
        for i, j in matching.items():
            print(f'{i}->{j}: {V[i][j]}')
        print(f'welfare = {weight} \n')
    return matching, weight


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


def elicitation_algorithm(V, num_samples, delta, c, excluded_pairs, noise_scale, flag_print_debug=False):
    """
    The Elicitation Algorithm
    :param V: the unit demand market's valuation matrix
    :param num_samples: number of samples
    :param delta: failure probability
    :param c: values range
    :param noise_scale:
    :param excluded_pairs: pairs to ommit for sampling
    :param flag_print_debug:
    :return:
    """

    samples = [add_noise(V, excluded_pairs, noise_scale) for _ in range(0, num_samples)]
    # print(f'samples = \n')
    # pprint.pprint(samples)
    empirical_market = sum(samples) / num_samples
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


def elicitation_with_pruning(V, sampling_schedule, delta_schedule, c, target_epsilon, noise_scale, flag_print_debug=True):
    """
    Elicitation Algorithm with pruning.
    :param V:
    :param sampling_schedule:
    :param delta_schedule:
    :param c:
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
    estimated_epsilons = {i: {j: c for j in range(0, np.size(V, 1))} for i in range(0, np.size(V, 0))}

    # We keep track of the total number of samples used
    total_num_samples = 0

    # We keep track of pruning at each step
    total_pruned = []

    # Loop through the sampling schedule and sampling deltas.
    for t, (curr_num_samples, cur_delta) in enumerate(zip(sampling_schedule, delta_schedule)):
        cur_empirical_market, cur_epsilon = elicitation_algorithm(V=V,
                                                                  num_samples=curr_num_samples,
                                                                  delta=cur_delta,
                                                                  c=c,
                                                                  excluded_pairs=prune_set,
                                                                  noise_scale=noise_scale,
                                                                  flag_print_debug=flag_print_debug)
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
                    print(f'Try pruning pair: {i},{j}: ', end='')
                if can_be_pruned(V=cur_empirical_market,
                                 full_empirical_welfare=full_empirical_welfare,
                                 consumer=i,
                                 good=j,
                                 epsilon=cur_epsilon):
                    prune_set.add((i, j))
                    active_set.remove((i, j))
                    cur_total_pruned += 1
                    if flag_print_debug:
                        print(f'++CAN BE PRUNED')
                elif flag_print_debug:
                    print(f'--CANNOT BE PRUNED')
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
