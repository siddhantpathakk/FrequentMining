import pandas as pd
import time

from apriori import AprioriAlgorithm
from preprocessor import preprocess
from colorama import Fore, Style


def apriori_for_dataset(dataset_name, dataset_metadata, verbose=False, minsup_val=None):
    """
    Runs Apriori algorithm on the given dataset. For each dataset, the algorithm is run for different minsup values.
    The results are saved in a csv file in the logs folder.
    
    Note: For the amazon-reviews dataset, the minsup value must be provided. For other datasets, the minsup values must be
    provided in the dataset_metadata dictionary.

    Args:
        dataset_name (str): Name of the dataset (amazon-reviews, groceries, movielens)
        dataset_metadata (dict): Dictionary containing metadata of all datasets
        verbose (bool, optional): Verbosity of output. Defaults to False.
        minsup_val (int, optional): Minimum support value(only if using amazon-reviews). Defaults to None.
    """
    dataset_meta = dataset_metadata[dataset_name]
    
    if dataset_name == 'amazon-reviews':
        assert minsup_val is not None, 'Please provide a minsup value for Amazon-Reviews dataset.'
        dataset_meta['minsup'] = [minsup_val]
    
    threshold_rating = dataset_meta['threshold_rating']
    
    total_time = 0
    total_timer = time.time()
    df_path = dataset_meta['path'] if isinstance(dataset_meta['path'], str) else dataset_meta['path'][0]

    
    for minsup in dataset_meta['minsup']:
        
        cur_stream = 0
        results = pd.DataFrame(columns=['item_sets', 'supp_count'])
        apriori = AprioriAlgorithm(minsup= minsup, verbose=verbose)
        
        print(Fore.WHITE + f'Running Apriori on',
            Fore.GREEN + f'{dataset_name}',
            Fore.WHITE + f':',
            Fore.GREEN + f'minsup = {minsup}')
        print(Fore.WHITE + "##"*40)
        
        for chunk in pd.read_csv(df_path, chunksize = dataset_meta['limit'], header=None):
            
            cur_stream += 1
            print(Fore.WHITE + f'Accessing chunk #{cur_stream} with {len(chunk)} values.')
            
            timer2 = time.time()
        
            if dataset_name == 'movielens':
                movie_transactions = preprocess(dataset_name=dataset_name,
                                                ratings=chunk,
                                                movies=pd.read_csv(dataset_meta['path'][1]),
                                                threshold_rating=threshold_rating)

            else:
                movie_transactions = preprocess(dataset_name=dataset_name, 
                                                df=chunk, 
                                                threshold_rating=threshold_rating)
                
        
            freq_item_sets = apriori.run(movie_transactions)
            print(Fore.WHITE + f'Num. of transactions: {len(movie_transactions)}')

            
            if len(freq_item_sets)>0:
                print(Fore.WHITE + f'Num. of freq itemsets in this chunk:',
                    Fore.GREEN + f'{len(freq_item_sets)}')
            else:
                print(Fore.WHITE + f'Num. of freq itemsets: {len(freq_item_sets)}')
            
            
            results = pd.concat([results, freq_item_sets], ignore_index=True).drop_duplicates()
            print(Fore.WHITE + f'Total num. of freq itemsets till now: {len(results)}')
            
            
            
            print(Fore.WHITE + f'Finished data-stream #{cur_stream} in',
                Fore.GREEN +  f'{round(time.time() - timer2, 1)} seconds.')

            print(Fore.WHITE + '--'*40)


        # display(results)
        results.to_csv(f'./logs/{dataset_name}/itemsets_df_minsup_{minsup}.csv', index=False)
        
        print(Fore.WHITE + f'Time taken for minsup {minsup} =', 
            Fore.GREEN + f'{round(time.time() - timer2, 1)} seconds.')
        
        print(Fore.WHITE + Style.DIM + f'Itemsets saved to ./logs/{dataset_name}/itemsets_df_minsup_{minsup}.csv')
        print(Fore.WHITE + '--'*40)
        print()
    
    
    total_time += time.time() - total_timer
        
    print(Fore.GREEN +
        f'Completed {dataset_name} fp in {total_time:.1f} seconds.')

    print(Fore.WHITE + "##"*40, '\n\n')