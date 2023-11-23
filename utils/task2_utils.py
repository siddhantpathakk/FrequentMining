import sys
from pyspark import SparkConf, SparkContext
import json
import itertools
import time
from itertools import chain


def generateFreqitem(basket, ck, support_threshold):
    # Function to generate frequent itemsets from candidata itemsets
    # inputs are basket, candidate sets and support_threshold
    C1 = [tuple(x) for x in ck]
    cnt = {}  # dict to store count for each candidate
    for i in basket:
        for c in C1:
            if (set(c).issubset(i)):
                if c in cnt:
                    cnt[c] += 1
                else:
                    cnt[c] = 1

    freq_item = []  # frequent item to extract items count>=support_threshold
    for key in cnt:
        if cnt[key] >= support_threshold:
            freq_item.append(key)
    return freq_item  # return frequent items from candidate set


def son2count_freq(basket, subsets):
    # make sure to convert elemnet inside list to a tuple before starting to create dict count
    C1 = [tuple(x) for x in subsets]
    cnt = {}
    for i in basket:
        for c in C1:
            if (set(c).issubset(i)):
                if c in cnt:
                    cnt[c] += 1
                else:
                    cnt[c] = 1
    frequency = []
    for key in cnt:
        # append(frequency to be (frequent items, count) pairs)
        frequency.append([key, cnt[key]])
    return frequency


def generateFreqitem(basket, ck, support_threshold):
    # Function to generate frequent itemsets from candidata itemsets
    # inputs are basket, candidate sets and support_threshold
    C1 = [tuple(x) for x in ck]
    cnt = {}  # dict to store count for each candidate
    for i in basket:
        for c in C1:
            if (set(c).issubset(i)):
                if c in cnt:
                    cnt[c] += 1
                else:
                    cnt[c] = 1

    freq_item = []  # frequent item to extract items count>=support_threshold
    for key in cnt:
        if cnt[key] >= support_threshold:
            freq_item.append(key)
    return freq_item  # return frequent items from candidate set


def son2count_freq(basket, subsets):
    # make sure to convert elemnet inside list to a tuple before starting to create dict count
    C1 = [tuple(x) for x in subsets]
    cnt = {}
    for i in basket:
        for c in C1:
            if (set(c).issubset(i)):
                if c in cnt:
                    cnt[c] += 1
                else:
                    cnt[c] = 1
    frequency = []
    for key in cnt:
        # append(frequency to be (frequent items, count) pairs)
        frequency.append([key, cnt[key]])
    return frequency


def apriori(basket, support, num_baskets):
    baskets = list(basket)
    # frequent itemsets to store from singles, pairs and triples,etc
    frequentitemsets = []
    # create empty dic to store singleton count
    c = {}
    # empty list to store freq singles/pairs/triples,etc.
    freq = []
    # define support threshold
    support_threshold = float(len(baskets)/num_baskets)*support
    k = 1
    # generating singletons
    for i in baskets:
        for j in i:
            if j in c:
                c[j] += 1
            else:
                c[j] = 1

    # count for frequent singles
    for key in c:
        if c[key] >= support_threshold:
            freq.append(key)

    freq.sort()
    # maker singletons to tuple format for future implementation
    freq1 = [(x,) for x in freq]
    # append frequent singles to the frequent itemsets
    frequentitemsets = freq1
    k += 1

    # generating pairs
    C = []
    pair = []
    for i in itertools.combinations(freq, 2):
        pair = list(i)
        C.append(pair)
    C.sort()
    # Again, make it to tuple form
    C1 = [tuple(l) for l in C]
    ck2 = {}
    for i in baskets:
        for c in C1:
            if (set(c).issubset(i)):
                if c in ck2:
                    ck2[c] += 1
                else:
                    ck2[c] = 1
    # Append frequent pairs to freq
    freq.clear()
    for key in ck2:
        if (ck2[key] >= support_threshold):
            freq.append(key)
    freq.sort()
    frequentitemsets.extend(freq)
    k += 1

# countinue to generate triples, quadraples etc in while loop, by increasing k one at a time.
    while len(C) > 0:
        C.clear()
        for i in range(len(freq)):
            for j in range(i+1, len(freq)):
                # reference
                list1 = list(freq[i])[:k-2]
                list2 = list(freq[j])[:k-2]
                if list1 == list2:
                    # set union to append elements
                    C.append(sorted(list(set(freq[i]).union(set(freq[j])))))
                else:
                    break
        C.sort()
        freq.clear()
        freq = generateFreqitem(baskets, C, support_threshold)
        freq.sort()
        frequentitemsets.extend(freq)
        k += 1

    return frequentitemsets
