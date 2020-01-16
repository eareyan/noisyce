
num_consu = 2
num_goods = 5
scale = 10

# the_V = uniform_random_distribution(num_consu, num_goods, scale)
# the_V = preferred_subset_distribution(num_consu, num_goods, scale)
the_V = preferred_good_distribution(num_consu, num_goods, scale)
print(f'the_V = \n{the_V}')

the_matching, the_welfare = get_maximum_welfare(V=the_V, flag_print_matching=True)

# Test solve for highest and lowest CE prices.

# Lowest CE
lowest_prices = solve_ce_prices_lp(V=the_V, matching=the_matching, minimize=True)
compute_market_regret(the_V, the_matching, lowest_prices)

# Highest CE
highest_prices = solve_ce_prices_lp(V=the_V, matching=the_matching, minimize=False)
compute_market_regret(the_V, the_matching, highest_prices)