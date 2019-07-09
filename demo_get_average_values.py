import statistics as stats
import operator


def get_average_values(dicts):
    keys = sorted(dicts[0].keys())
    mean_values = [round(stats.mean([d[k] for d in dicts]), 2) for k in keys]
    return dict(zip(keys, mean_values))


def get_max(dicts):
    return dict([max(dicts.items(), key=operator.itemgetter(1))])


d1 = {'B': 200, 'A': 100, 'C': 150}
d2 = {'A': 80, 'B': 195, 'C': 170}
d3 = {'A': 95, 'B': 212, 'C': 140}

dictionaries = [d1, d2, d3]

dictionaries_average = get_average_values(dictionaries)

print(f"average values: {dictionaries_average}")

print("entry with max value: {}".format(get_max(dictionaries_average)))
