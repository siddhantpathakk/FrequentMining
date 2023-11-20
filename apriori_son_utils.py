import time
from apriori_utils import generateFrequentSingletons, countFrequentItemsets, generateKTupleItemsets

def turnStr2Pair(pairStr):
    return (pairStr.split(',')[0], pairStr.split(',')[1])


def apriori_partition(chunk, threshold, full_size):
    # chunk_list is needed here because chunk is of type TraversableOnce, which will be empty after calling list
    chunk_list = list(chunk)
    chunk_size = len(chunk_list)
    # determine local(chunk) threshold, 0.9 is the scaling factor used to reduce false positives
    local_threshold = (chunk_size/full_size) * threshold
    if local_threshold < 1:
        local_threshold = 1

    result_candidate_itemsets = list()  # contains all k-tuple itemsets as result
    # contains the true frequent itemsets used to generate candidate for next k in apriori
    true_frequent_k_itemset_list = list()
    # contains candidate of next k-tuple itemsets, used to decide if there are any candidate left
    next_k_candidate_list = list()
    k_index = 1

    # generate true frequent singletons
    # true_frequent_k_itemset_list is now the frequent singletons
    # next_k_candidate_list is now the singleton candidates

    genSingleStart = time.time()
    true_frequent_k_itemset_list, next_k_candidate_list = generateFrequentSingletons(
        chunk_list, local_threshold)

    while len(next_k_candidate_list) != 0:
        k_index += 1  # generate 2-tuple candidates, 3-tuple, 4-tuple, etc

        if k_index == 2:

            result_candidate_itemsets.append(
                # append candidate k
                [(single,) for single in true_frequent_k_itemset_list])
            next_k_candidate_list = generateKTupleItemsets(
                # true k-1 to candidate k
                [(single,) for single in true_frequent_k_itemset_list], 2)
        else:
            true_frequent_k_itemset_list = countFrequentItemsets(
                next_k_candidate_list, chunk_list, local_threshold)  # cand k to true k
            result_candidate_itemsets.append(
                true_frequent_k_itemset_list)  # append candidate k
            next_k_candidate_list = generateKTupleItemsets(
                true_frequent_k_itemset_list, k_index)  # true k to candidate k+1

    yield result_candidate_itemsets


def countItemsets(cand, busResult):
    # busResult is list of set
    count = 0
    for bus in busResult:
        if set(cand).issubset(bus):
            count += 1
    return (cand, count)

def get_dataset_info(dataset_name):
    _dataset_metadata = {
        'groceries': {
            'path': r'./data/groceries/transactions.csv',
            'minsup': 50,
            'minsup_step': 50,
            'partition_num': 10,
        },

        'movielens': {
            'path': r'./data/movielens/transactions.csv',
            'minsup': 100,
            'minsup_step': 100,
            'partition_num': 100
        },

        'yelp': {
            'path': r'./data/yelp/transactions.csv',
            'minsup': 100,
            'minsup_step': 100,
            'partition_num': 1000
        }
    }
    
    return _dataset_metadata[dataset_name]


