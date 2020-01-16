import numpy as np


def uniform_random_distribution(num_consumer, num_items, scale):
    """
    Generate a market where all valuations are i.i.d. uniform (0, 1)*scale
    :param num_consumer:
    :param num_items:
    :param scale:
    :return:
    """
    return np.random.rand(num_consumer, num_items) * scale


def preferred_good_distribution(num_consumers, num_items, scale):
    """
    Generate a market from the preferred_good distribution.
    :param num_consumers:
    :param num_items:
    :return:
    """
    preferred_goods = [np.random.randint(0, num_items - 1) for _ in range(0, num_consumers)]
    preferred_goods_values = np.random.rand(num_consumers) * scale

    return np.array([[preferred_goods_values[i] / (2 ** (j + 2)) if j != preferred_goods[i] else preferred_goods_values[i] for j in range(0, num_items)]
                     for i in range(0, num_consumers)])


def preferred_subset_distribution(num_consumers, num_items, scale):
    """
    Generate a market from the preferred_subset distribution.
    :param num_consumers:
    :param num_items:
    :param scale:
    :return:
    """
    preferred_sets = [np.random.choice(num_items, np.random.randint(1, num_items)) for _ in range(0, num_consumers)]
    return np.array([[np.random.rand() * scale if j in preferred_sets[i] else 0.0 for j in range(0, num_items)] for i in range(0, num_consumers)])
