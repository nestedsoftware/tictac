import statistics as stats
import operator


def get_average_values(dicts):
    keys = dicts[0].keys()
    key_avg_value_pairs = [(k, round(stats.mean([d[k] for d in dicts]), 2))
                           for k in keys]
    return dict(key_avg_value_pairs)


def get_max(dicts):
    return dict([max(dicts.items(), key=operator.itemgetter(1))])


d1 = {'A': 100, 'B': 200, 'C': 150}
d2 = {'A': 80, 'B': 195, 'C': 170}
d3 = {'A': 95, 'B': 212, 'C': 140}

dictionaries = [d1, d2, d3]

dictionaries_average = get_average_values(dictionaries)

print(f"average values: {dictionaries_average}")

print("entry with max value: {}".format(get_max(dictionaries_average)))
