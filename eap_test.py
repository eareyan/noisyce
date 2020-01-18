from unit_demand import elicitation_with_pruning, epsilon_to_num_samples
from unit_demand_generator import uniform_random_distribution, preferred_subset_distribution
import unit_demand_experiments_config as config
import math
import time

t0 = time.time()
# Parameters
num_consumers = 5
num_goods = 5
noise_factor = 1.0
type_market = uniform_random_distribution
# type_market = preferred_subset_distribution
target_eps = 0.05

# Derived Parameters
c = config.values_high - config.values_low + noise_factor
num_samples_ea = epsilon_to_num_samples(num_consumers, num_goods, target_eps, config.delta, c, set())
sampling_schedule = [math.ceil(num_samples_ea * factor) for factor in config.doubling_schedule]
delta_schedule = [config.delta / len(config.doubling_schedule) for _ in range(0, len(config.doubling_schedule))]

# Draw a random market
market = type_market(num_consumers, num_goods, config.values_high, config.values_low)
# Run EAP experiment.
_, _, final_delta, final_epsilon, total_num_samples, total_pruned = elicitation_with_pruning(V=market,
                                                                                             sampling_schedule=sampling_schedule,
                                                                                             delta_schedule=delta_schedule,
                                                                                             values_high=config.values_high,
                                                                                             values_low=config.values_low,
                                                                                             noise_factor=noise_factor,
                                                                                             target_epsilon=0.0001,
                                                                                             flag_print_debug=False)
# Print parameters
print(f'sampling_schedule = {sampling_schedule}')
print(f'delta_schedule = {delta_schedule}')
# Print results
num_samples_ea_cf = epsilon_to_num_samples(num_consumers, num_goods, final_epsilon, config.delta, c, set()) * num_consumers * num_goods
print(f'num_samples_ea_cf = {num_samples_ea_cf}, eap_samples = {total_num_samples}')
print(f'savings = {(1.0 - (total_num_samples / num_samples_ea_cf)) * 100.0:.2f}%')
print(f'total_pruned = {total_pruned}')
print(f'took {time.time() - t0:.2f} sec')
