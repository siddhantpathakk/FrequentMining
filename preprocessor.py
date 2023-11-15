import pandas as pd

DATASETS = {
    'groceries' : './data/groceries/groceries.csv',
    'movielens' : {
        'ratings' : './data/movielens/ratings.csv',
        'movies' : './data/movielens/movies.csv',
    },
    'amazon-reviews' : './data/amazon-reviews/all_csv_files.csv'
}

class Preprocessor:
    def __init__(self):
        pass
    
    def load_data(self, dataset_name, threshold=4):
        if dataset_name == 'groceries':
            return self.load_groceries()
        elif dataset_name == 'movielens':
            return self.load_movielens(threshold=threshold)
        elif dataset_name == 'amazon-reviews':
            return self.load_amazon_reviews()
        else:
            raise Exception('Dataset not found')
        
    def load_groceries(self):
        groceries_df = pd.read_csv(DATASETS['groceries'])
        groceries_transactions = groceries_df.groupby(
            'Member_number')['itemDescription'].apply(list).reset_index()

        # Convert the lists in the "itemDescription" column to sets
        groceries_transactions["itemDescription"] = groceries_transactions["itemDescription"].apply(
            set)
        
        print(groceries_transactions.head())
        
        # Convert the transactions into a list of lists
        transactions_list = groceries_transactions['itemDescription'].tolist()

        groceries_transactions["itemDescription"] = groceries_transactions["itemDescription"].agg(
            lambda x: ",".join(x))


        groceries_transactions = groceries_transactions["itemDescription"].str.split(
            ',')
        
        return groceries_transactions
    
    def load_movielens(self, threshold=4):
        ratings = pd.read_csv(DATASETS['movielens']['ratings'])
        movies = pd.read_csv(DATASETS['movielens']['movies'])
        
        # Filter ratings based on the threshold
        filtered_ratings = ratings[ratings['rating'] >= threshold]

        # Merge the filtered ratings with the movies dataframe to get the movie names associated with the movie IDs.
        merged_data = pd.merge(filtered_ratings, movies, on='movieId', how='inner')

        # Group the data by user ID and collect movie titles as items in transactions
        transactions = merged_data.groupby('userId')['title'].apply(list).reset_index()

        transactions = transactions.set_index('userId')

        # Convert sets to a single string with items separated by a comma
        transactions["title"] = transactions["title"].agg(lambda x: ",".join(x))

        movie_transactions = transactions["title"].str.split(',')

        return movie_transactions
    
    def load_amazon_reviews(self, threshold=4):
        df = pd.read_csv('../data/amazon-reviews/all_csv_files.csv', nrows=10000, header=None)
        df.columns = ['reviewer_id', 'item_id', 'rating', 'timestamp']
        threshold_rating = threshold
        
        filtered_ratings = df[df['rating'] >= threshold_rating]

        transactions = filtered_ratings.groupby('reviewer_id')['item_id'].apply(
            list).reset_index().set_index('reviewer_id')

        # Convert the transactions into a list of lists
        transactions_list = transactions['item_id'].tolist()


        # Convert sets to a single string with items separated by a comma
        transactions["item_id"] = transactions["item_id"].agg(lambda x: ",".join(x))

        df_transactions = transactions["item_id"].str.split(',')
        
        return df_transactions
