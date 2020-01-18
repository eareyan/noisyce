import numpy as np


def uniform_random_distribution(num_consumers, num_goods, values_high, values_low):
    """
    Generate a market where all valuations are i.i.d. uniform (0, 1)*scale
    :param num_consumers:
    :param num_goods:
    :param values_high:
    :param values_low:
    :return:
    """
    return (values_high - values_low) * np.random.rand(num_consumers, num_goods) + values_low


def preferred_good_distribution(num_consumers, num_goods, values_high, values_low, distinct=False):
    """
    Generate a market from the preferred_good distribution.
    :param num_consumers:
    :param num_goods:
    :param values_high:
    :param values_low:
    :return:
    """
    preferred_goods = np.random.choice(num_goods, num_consumers, replace=not distinct)
    preferred_goods_values = (values_high - values_low) * np.random.rand(num_consumers) + values_low

    return np.array([[preferred_goods_values[i] / (2 ** (j + 1)) if j != preferred_goods[i] else preferred_goods_values[i] for j in range(0, num_goods)]
                     for i in range(0, num_consumers)])


def preferred_distinct_good_distribution(num_consumers, num_goods, values_high, values_low):
    return preferred_good_distribution(num_consumers, num_goods, values_high, values_low, True)


def preferred_subset_distribution(num_consumers, num_goods, values_high, values_low):
    """
    Generate a market from the preferred_subset distribution.
    :param num_consumers:
    :param num_goods:
    :param values_high:
    :param values_low:
    :return:
    """
    preferred_sets = [np.random.choice(num_goods, np.random.randint(1, num_goods)) for _ in range(0, num_consumers)]
    return np.array([[(values_high - values_low) * np.random.rand() + values_low if j in preferred_sets[i] else 0.0
                      for j in range(0, num_goods)] for i in range(0, num_consumers)])
