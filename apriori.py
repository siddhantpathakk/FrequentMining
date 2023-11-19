import pandas as pd
from preprocessor import preprocess

def prune(data,supp):
    df = data[data['supp_count'] >= supp]
    return df

def count_itemset(transaction_df, itemsets):
    count_item = {}
    for item_set in itemsets:
        set_A = set(item_set)
        for row in transaction_df:
            set_B = set(row)

            if set_B.intersection(set_A) == set_A:
                if item_set in count_item.keys():
                    count_item[item_set] += 1

                else:
                    count_item[item_set] = 1

    data = pd.DataFrame()
    data['item_sets'] = count_item.keys()
    data['supp_count'] = count_item.values()
    # print("Candidate itemset table (Counting):\n", data)

    return data

def count_item(trans_items):
    count_ind_item = {}
    for row in trans_items:
        for i in range(len(row)):
            if row[i] in count_ind_item.keys():
                count_ind_item[row[i]] += 1
            else:
                count_ind_item[row[i]] = 1

    data = pd.DataFrame()
    data['item_sets'] = count_ind_item.keys()
    data['supp_count'] = count_ind_item.values()
    data = data.sort_values('item_sets')

    return data

def join(list_of_items):
    itemsets = []
    i = 1
    for entry in list_of_items:
        proceding_items = list_of_items[i:]
        for item in proceding_items:
            if(type(item) is str):
                if entry != item:
                    tuples = (entry, item)
                    itemsets.append(tuples)
            else:
                if entry[0:-1] == item[0:-1]:
                    tuples = entry+item[1:]
                    itemsets.append(tuples)
        i = i+1
    if(len(itemsets) == 0):
        return None

    return itemsets

def apriori(trans_data,supp=10):
    freq = pd.DataFrame()

    df = count_item(trans_data)

    while(len(df) != 0):

        df = prune(df, supp)
        # print("Minsup =", supp,"\n")
        # print("Freq itemset table (Pruned):\n", df)

        if len(df) > 1 or (len(df) == 1 and int(df.supp_count >= supp)):
            freq = df

        itemsets = join(df.item_sets)

        if(itemsets is None):
            return freq

        df = count_itemset(trans_data, itemsets)
        
    return freq


def _convert_apriori_to_set(apriori_fi_df):
    freq_itemsets = set()
    for _ , row in apriori_fi_df.iterrows():
        freq_itemsets.add(frozenset(row['item_sets']))
    return freq_itemsets


def freq_itemset_partition_db(dataset_csv_file, num_partitions, min_supp, preprocess_dataset=True, **kwargs):
    partition_itemsets = {}
    for trans_df in pd.read_csv(dataset_csv_file, chunksize=num_partitions, low_memory=True, engine='c', **kwargs):
        
        if preprocess_dataset:
            trans_df = preprocess(trans_df)
            
        df = count_item(trans_df)
        while(len(df) != 0):
            df = prune(trans_df, min_supp)
            itemsets = join(df.item_sets)

            if(itemsets is None):
                break
            df = count_itemset(trans_df, itemsets)
            
            for i in range(len(df)):
                if df.item_sets[i] not in partition_itemsets.keys():
                    partition_itemsets[df.item_sets[i]] = df.supp_count[i]
                else:
                    partition_itemsets[df.item_sets[i]] += df.supp_count[i]
                
    return partition_itemsets