import pprint
import time
import numpy as np
from unit_demand import elicitation_with_pruning, get_maximum_welfare, epsilon_to_num_samples
from unit_demand_generator import uniform_random_distribution, preferred_good_distribution, preferred_subset_distribution


def run_test(V, V_shifted, sampling_schedule, delta_schedule, c, target_epsilon, noise_scale):
    t0 = time.time()
    final_estimated_values, final_estimated_epsilons, final_delta, final_epsilon, total_num_samples, num_pruned = elicitation_with_pruning(V=V,
                                                                                                                                           sampling_schedule=sampling_schedule,
                                                                                                                                           delta_schedule=delta_schedule,
                                                                                                                                           c=c,
                                                                                                                                           target_epsilon=target_epsilon,
                                                                                                                                           noise_scale=noise_scale,
                                                                                                                                           flag_print_debug=False)
    print(f'\nEnd Of Experiment, took {time.time() - t0} sec')
    """print(f'final_estimated_values = \n')
    pprint.pprint(final_estimated_values)
    print(f'final_estimated_epsilons = \n')
    pprint.pprint(final_estimated_epsilons)"""
    final_market = np.array([[final_estimated_values[i][j] for j in range(0, np.size(V, 1))] for i in range(0, np.size(V, 0))])
    # print(f'estimated final_market = {final_market}')

    _, estimated_welfare = get_maximum_welfare(V=final_market, flag_print_matching=False)
    _, true_welfare = get_maximum_welfare(V=V_shifted, flag_print_matching=False)
    print(f'estimated_welfare = {estimated_welfare}, true_welfare = {true_welfare}')
    print(f'diff = {abs(true_welfare - estimated_welfare)}, eps = {final_epsilon}, {2*np.size(V, 0)*final_epsilon >= abs(true_welfare - estimated_welfare)}')

    print(f'\ntotal_num_samples = {total_num_samples}, num_pruned = {num_pruned}')
    print(f'final epsilon = {final_epsilon}')
    print(f'final delta = {final_delta}')
    ea_needed_samples = epsilon_to_num_samples(np.size(V, 0), np.size(V, 1), final_epsilon, final_delta, c, set()) * np.size(V, 0) * np.size(V, 1)
    print(f'samples needed by EA = {ea_needed_samples}, savings = {(1.0 - (total_num_samples/ea_needed_samples)) * 100.0:.2f}%')
    return total_num_samples


# Parameters
scale = 10.0
noise_scale = 1.0
# noise_scale = 2.5
# noise_scale = 5.0
# noise_scale = 10.0
# noise_scale = 100.0
the_c = scale + noise_scale
the_target_epsilon = 0.0001

# Some example markets
# the_V = uniform_random_distribution(10, 10, scale)
# the_V = preferred_good_distribution(5, 15, scale)
the_V = preferred_subset_distribution(10, 10, scale)
the_V_shifted = the_V + (noise_scale / 2.0)  # assumes uniform noise
print(f"V_shifted =\n {the_V_shifted}")
print(f"V =\n {the_V}")

# Some example schedules.
# the_sampling_schedule = [10000, 40000, 80000]
# the_sampling_schedule = [11497, 22993, 45986]
the_sampling_schedule = [10884, 21767, 43533]
# the_bad_schedule = [1000, 15000, 30000]
the_delta_schedule = [0.05, 0.05, 0.05]
delta = sum(the_delta_schedule)

print(f'num_consumers = {10}, \n'
      f'num_goods = {10}, \n'
      f'type_market = uniform_random_distribution, \n'
      # f'target_eps = {target_eps}, \n'
      f'scale = {scale}, \n'
      f'noise_scale = {noise_scale}, \n'
      # f'num_samples_ea = {num_samples_ea* num_consumers * num_goods }, \n'
      f'sampling_schedule = {the_sampling_schedule}, \n'
      f'delta_schedule = {the_delta_schedule} \n\n')

# Start the test
run_test(the_V, the_V_shifted, the_sampling_schedule, the_delta_schedule, the_c, the_target_epsilon, noise_scale)
# run_test(the_V, the_V_shifted, the_bad_schedule, the_delta_schedule, the_c, the_target_epsilon, noise_scale)
