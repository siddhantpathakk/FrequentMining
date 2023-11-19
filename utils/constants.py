dataset_metadata = {
    'amazon-reviews': {
        'path': './data/amazon-reviews/all_csv_files.csv',
        'size': 233055326,    
        'limit': 10**5,
        'minsup': [1000, 1500, 3000, 5000],
        'threshold_rating': 4.0
        
    },
    
    
    'groceries': {
        'path': './data/groceries/Groceries_dataset.csv',
        'size': 38766,
        'limit': 3000,
        'minsup': [50, 70, 150, 200],
        'threshold_rating': 4.0
    },
    
    
    'movielens': {
        'path': ['./data/movielens/ratings.csv', './data/movielens/movies.csv'],
        'size': 100836, 
        'limit': 10000,
        'minsup': [50, 100, 150, 200, 250],
        'threshold_rating': 4.0
    }
}
