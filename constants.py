dataset_metadata = {
    'amazon-reviews': {
        'path': './data/amazon-reviews/all_csv_files.csv',
        
        # using only half the dataset
        'size': 233055326//2,  # original size = 233055326 
        
        'limit': 10**5,
        'minsup': [1500, 1000, 500, 250],
        'threshold_rating': 4.0
        
    },
    
    'groceries': {
        'path': './data/groceries/Groceries_dataset.csv',
        'size': 38766,
        'limit': 10000,
        'minsup': [10, 50, 100, 150, 200],
        'threshold_rating': 4.0
    },
    
    
    'movielens': {
        'path': ['./data/movielens/ratings.csv', './data/movielens/movies.csv'],
        'size': 100836, 
        'limit': 50000,
        'minsup': [50, 100, 150, 200, 250],
        'threshold_rating': 4.0
    }
}

verbose = False