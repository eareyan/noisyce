from unit_demand_generator import uniform_random_distribution, preferred_good_distribution, preferred_subset_distribution

# Common parameters for experiments.
num_trials = 20
scale = 10.0
delta = 0.1
num_consumers = [5, 10, 15, 20]
num_goods = [5, 10, 15, 20]
type_of_markets = [uniform_random_distribution, preferred_good_distribution, preferred_subset_distribution]
noise_scale = [1.0, 2.5, 5.0]
epsilons = [0.125, 0.25, 0.5, 1.0]
