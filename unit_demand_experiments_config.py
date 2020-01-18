from unit_demand_generator import uniform_random_distribution, preferred_good_distribution, preferred_distinct_good_distribution, preferred_subset_distribution

# Discarded Parameters
# epsilons = [0.1, 0.05, 0.025] # candidate, but 0.025 takes too long.


# Common parameters for experiments.
num_trials = 15
values_high = 10.0
values_low = 0.0
delta = 0.1

# Candidate Parameters
noise_factor = [1.0, 2.0, 4.0]
epsilons = [0.2, 0.15, 0.1, 0.05]
type_of_markets = [uniform_random_distribution,
                   preferred_good_distribution,
                   preferred_distinct_good_distribution,
                   preferred_subset_distribution]
num_consumers = [5, 10, 15, 20]
num_goods = [5, 10, 15, 20]

# EAP
doubling_schedule = [0.25, 0.5, 1.0, 2.0]
