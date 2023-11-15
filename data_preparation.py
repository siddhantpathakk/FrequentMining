import pandas as pd
from itertools import combinations

def prune(data, supp):
  """
  pruning to get Candidates k-itemset to Frequent k-itemset
  by comparing the support count (data.supp_count) with the minSup (supp)
  """
  df = data[data.supp_count >= supp]
  return df


def count_itemset(transaction_df, itemsets):
  """
  for each itemset in (the current set of) freq-n itemsets,
  the count of each itemset is computed according to
  the number of occurrences that it appears in the transactions data
  """
  count_item = {}
  for item_set in itemsets:
      # set A represents the itemset whose count is to be computed in order to be determined if it is a frequent itemset or not
      set_A = set(item_set)
      for row in transaction_df:
          set_B = set(row)  # set B represents the transaction row record
          # checks for occurrence of the itemset in the transaction
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
            if (type(item) is str):
                if entry != item:
                    tuples = (entry, item)
                    itemsets.append(tuples)
            else:
                if entry[0:-1] == item[0:-1]:
                    tuples = entry+item[1:]
                    itemsets.append(tuples)
        i = i+1
    if (len(itemsets) == 0):
        return None
    return itemsets


def convert_to_transdf(data):
    # Create a set of all unique items
    all_items = set()
    for entry in data:
      print("entry:", entry)
      all_items.update(entry["items"])

    # Generate all possible combinations of items
    combinations_list = []
    for r in range(1, len(all_items) + 1):
      item_combinations = combinations(all_items, r)
      combinations_list.extend(item_combinations)

    # Create a DataFrame with features for each combination
    df = pd.DataFrame(data)
    for combination in combinations_list:
      feature_name = " & ".join(sorted(list(combination)))
      df[feature_name] = df["items"].apply(
          lambda x: int(set(combination).issubset(x)))

    # Fill NaN values with 0
    df.fillna(0, inplace=True)

    # Set the "record_id" as the DataFrame index
    df.set_index("record_id", inplace=True)

    # Print the resulting DataFrame
    print(df)

    return df
