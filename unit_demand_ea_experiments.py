import itertools as it
from unit_demand import epsilon_to_num_samples, elicitation_algorithm, get_maximum_welfare
import unit_demand_experiments_config as config
from unit_demand_ce_solver import solve_ce_prices_lp, compute_market_regret
import pandas as pd


def save_results(results_data):
    print('\n Saving results...')
    results_dataframe = pd.DataFrame(results_data,
                                     columns=['num_consumers',
                                              'num_goods',
                                              'type_market',
                                              'eps',
                                              'num_samples',
                                              'noise_scale',
                                              'gt_welfare',
                                              'emp_welfare',
                                              'max_regret_low_prices_first',
                                              'max_regret_high_prices_first',
                                              'max_regret_low_prices_second',
                                              'max_regret_high_prices_second'])
    results_dataframe.to_csv('results.csv', index=False)


def check_regrets(left_market, right_market):
    # Welfare
    left_matching, left_welfare = get_maximum_welfare(V=left_market, flag_print_matching=False)
    # Lowest CE for the gt market
    left_lowest_prices = solve_ce_prices_lp(V=left_market, matching=left_matching, minimize=True)
    # Highest CE
    left_highest_prices = solve_ce_prices_lp(V=left_market, matching=left_matching, minimize=False)

    # Compute regret for gt low prices CE in the estimated market
    max_reg_low_prices = compute_market_regret(right_market, left_matching, left_lowest_prices)
    # Compute regret for gt high prices CE in the estimated market
    max_reg_high_prices = compute_market_regret(right_market, left_matching, left_highest_prices)

    return left_welfare, max_reg_low_prices, max_reg_high_prices


total = 0
results = []
for num_consumers, num_goods, type_market, noise_scale, eps in it.product(config.num_consumers,
                                                                          config.num_goods,
                                                                          config.type_of_markets,
                                                                          config.noise_scale,
                                                                          config.epsilons):
    c = config.scale + noise_scale
    num_samples = epsilon_to_num_samples(num_consumers, num_goods, eps, config.delta, c, set())
    print('\n', num_consumers, num_goods, type_market.__name__, noise_scale, eps, num_samples)
    for trial in range(0, config.num_trials):
        print(f'#{trial}', end=' ')
        # Get a random market
        market = type_market(num_consumers, num_goods, noise_scale)

        # Run the Elicitation Algorithm
        empirical_market, epsilon = elicitation_algorithm(market, num_samples, config.delta, c, set(), noise_scale)

        # The ground truth market is a shifted Market
        gt_market = market + (noise_scale / 2.0)  # assumes uniform noise
        """
        print(market)
        print(empirical_market)
        print(gt_market)
        """

        # Check First containment
        gt_welfare, max_regret_low_prices_first, max_regret_high_prices_first = check_regrets(gt_market, empirical_market)

        # Check First containment
        emp_welfare, max_regret_low_prices_second, max_regret_high_prices_second = check_regrets(empirical_market, gt_market)

        # Print Debug
        """
        print(f'eps = {epsilon}')
        print(f'max_regret_low_prices_first = {max_regret_low_prices_first}')
        print(f'max_regret_high_prices_first = {max_regret_high_prices_first}')
        print(f'max_regret_low_prices_second = {max_regret_low_prices_second}')
        print(f'max_regret_high_prices_second = {max_regret_high_prices_second}')
        """

        results += [(num_consumers,
                     num_goods,
                     type_market.__name__,
                     eps,
                     num_samples,
                     noise_scale,
                     gt_welfare,
                     emp_welfare,
                     max_regret_low_prices_first,
                     max_regret_high_prices_first,
                     max_regret_low_prices_second,
                     max_regret_high_prices_second)]

        total += 1
    save_results(results)
    # break

save_results(results)
