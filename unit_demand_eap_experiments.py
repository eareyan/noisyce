import unit_demand_experiments_config as config
import itertools as it
from unit_demand import elicitation_with_pruning, epsilon_to_num_samples
import math
import time
import pandas as pd


def save_results(results_data):
    print('\n Saving results...')
    results_dataframe = pd.DataFrame(results_data, columns=['num_consumers',
                                                            'num_goods',
                                                            'type_market',
                                                            'noise_factor',
                                                            'target_eps',
                                                            'final_epsilon',
                                                            'total_num_samples']
                                                           + [f'total_pruned_{i}' for i in range(0, len(config.doubling_schedule) - 1)])
    results_dataframe.to_csv('results_eap.csv', index=False)


total = 0
results = []
for num_consumers, num_goods, type_market, noise_factor, target_eps in it.product(config.num_consumers,
                                                                                  config.num_goods,
                                                                                  config.type_of_markets,
                                                                                  config.noise_factor,
                                                                                  config.epsilons):
    # Compute c
    c = config.values_high - config.values_low + noise_factor
    # Compute the number of samples that EA needs to achieve the target epsilon
    num_samples_ea = epsilon_to_num_samples(num_consumers, num_goods, target_eps, config.delta, c, set())
    # Set up the doubling schedule
    sampling_schedule = [math.ceil(num_samples_ea * factor) for factor in config.doubling_schedule]
    # Set up the delta schedule
    delta_schedule = [config.delta / len(config.doubling_schedule) for _ in range(0, len(config.doubling_schedule))]

    # Debug Prints
    """
    print(f'num_consumers = {num_consumers}, \n'
          f'num_goods = {num_goods}, \n'
          f'type_market = {type_market.__name__}, \n'
          f'target_eps = {target_eps}, \n'
          f'scale = {config.scale}, \n'
          f'noise_factor = {noise_factor}, \n'
          f'num_samples_ea = {num_samples_ea* num_consumers * num_goods }, \n'
          f'sampling_schedule = {sampling_schedule}, \n'
          f'delta_schedule = {delta_schedule} \n\n')"""

    print('\n', num_consumers, num_goods, type_market.__name__, noise_factor, target_eps)
    t0 = time.time()
    for trial in range(0, config.num_trials):
        print(f'#{trial}', end=' ')
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
        # Debug Prints
        """
        print(f'final_epsilon = {final_epsilon}, \n'
              f'final_delta = {final_delta}, \n'
              f'total_num_samples = {total_num_samples}, \n'
              f'num_pruned = {num_pruned}, \n'
              f'savings = {(1.0 - (total_num_samples / (epsilon_to_num_samples(num_consumers, num_goods, final_epsilon, final_delta, c, set()) * num_consumers * num_goods))) * 100.0:.2f}% \n'
              f'took = {time.time() - t0} sec. \n\n')
              """
        # Collect Results
        results += [(num_consumers,
                     num_goods,
                     type_market.__name__,
                     noise_factor,
                     target_eps,
                     final_epsilon,
                     total_num_samples)
                    + tuple(total_pruned[i] for i in range(0, len(config.doubling_schedule) - 1))]

    save_results(results)
