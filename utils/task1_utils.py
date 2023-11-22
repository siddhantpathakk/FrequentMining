import collections

def generateFrequentSingletons(chunk_list, local_minsup):
    counter_dict = dict()
    frequent_singletons = list()
    candidate = list()

    for basket in chunk_list:
        for item in basket:
            if item not in counter_dict.keys():
                counter_dict[item] = 1
            else:
                counter_dict[item] += 1
    candidate = counter_dict.keys()

    for key, value in counter_dict.items():
        if value >= local_minsup:
            frequent_singletons.append(key)
    return frequent_singletons, sorted(candidate)


def countFrequentItemsets(candidate_list, chunk_list, minsup):
    counter_dict = collections.defaultdict(int)
    # chunk_list is list of set
    for basket in chunk_list:
        for candidate in candidate_list:
            if set(candidate).issubset(basket):
                counter_dict[candidate] += 1

    frequent_dict = dict(
        filter(lambda pair: pair[1] >= minsup, counter_dict.items()))

    return list(frequent_dict.keys())  # list of tuple


def generateKTupleItemsets(frequent_list, k):
    candidatesList = set()
    for i in range(len(frequent_list)):
        for j in range(i+1, len(frequent_list)):
            item1 = frequent_list[i]
            item2 = frequent_list[j]
            cand = set(item1).union(set(item2))
            if len(cand) == k:
                candidatesList.add(tuple(sorted(cand)))
    return candidatesList