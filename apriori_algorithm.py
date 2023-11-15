import pandas as pd
from data_preparation import prune, count_itemset, count_item, join

class AprioriAlgorithm:
    def __init__(self, minsup, verbose=False):
        self.supp = minsup
        self.verbose = verbose
        self.freq_item_sets = None
    
    def run(self, trans_data):
        freq = pd.DataFrame()

        df = count_item(trans_data)  # to generate counts of
        
        if self.verbose:
            print(f'Dataset size: {len(df)}')
            
        while (len(df) != 0):

            df = prune(df, self.supp)
            
            if self.verbose:
                print("Minsup =", self.supp,"\n")
                print("Freq itemset table (Pruned):\n", df)

            if len(df) > 1 or (len(df) == 1 and int(df.supp_count >= self.supp)):
                freq = df

            itemsets = join(df.item_sets)

            if (itemsets is None):
                return freq

            df = count_itemset(trans_data, itemsets)
        
        self.freq_item_sets = df
        return df

    def transform_data(self, freq_item_sets):
        if freq_item_sets is None:
            freq_item_sets = self.freq_item_sets
        input_data = dict(freq_item_sets['item_sets'])
        transformed_data = []
        record_id = 1
        
        for _, items_tuple in input_data.items():
            items_set = set(items_tuple)
            transformed_data.append({"record_id": record_id, "items": items_set})
            record_id += 1
        
        if self.verbose:
            print(f'Transformed Data:\t',transformed_data)
            
        return transformed_data

