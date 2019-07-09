import statistics as stats


def get_average_values(dicts):
    keys = dicts[0].keys()
    key_avg_value_pairs = [(k, round(stats.mean([d[k] for d in dicts]), 2))
                           for k in keys]
    return dict(key_avg_value_pairs)


d1 = {'A': 100, 'B': 200, 'C': 150}
d2 = {'A': 80, 'B': 195, 'C': 170}
d3 = {'A': 95, 'B': 212, 'C': 140}

dictionaries = [d1, d2, d3]

print(get_average_values(dictionaries))
