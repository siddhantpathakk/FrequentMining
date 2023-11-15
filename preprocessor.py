import pandas as pd

class Streamer:
    def __init__(self, filepath, stream_limit):
        self.filepath = filepath
        self.stream_limit = stream_limit

    def getCurrentStream(self, stream_id):
        print(f'Accessing data-stream #{stream_id}')
        return pd.read_csv(self.filepath,
                           skiprows=stream_id*self.stream_limit,
                           nrows=self.stream_limit,
                           header=None)
        

def preprocess(dataset_name, df, threshold_rating, ratings=None, movies=None):

    if dataset_name == 'amazon-reviews':
        df.columns = ['reviewer_id', 'item_id', 'rating', 'timestamp']
        filtered_ratings = df[df['rating'] >= threshold_rating]
        transactions = filtered_ratings.groupby('reviewer_id')['item_id'].apply(
            list).reset_index().set_index('reviewer_id')
        transactions["item_id"] = transactions["item_id"].agg(
            lambda x: ",".join(x))
        movie_transactions = transactions["item_id"].str.split(',')
        return movie_transactions

    elif dataset_name == 'groceries':
        df.columns = ['Member_number', 'Date', 'itemDescription']
        groceries_transactions = df.groupby('Member_number')[
            'itemDescription'].apply(list).reset_index()
        groceries_transactions["itemDescription"] = groceries_transactions["itemDescription"].apply(
            set)
        # transactions_list = groceries_transactions['itemDescription'].tolist()
        groceries_transactions["itemDescription"] = groceries_transactions["itemDescription"].agg(
            lambda x: ",".join(x))
        groceries_transactions = groceries_transactions["itemDescription"].str.split(
            ',')
        return groceries_transactions

    elif dataset_name == 'movielens':
        assert ratings is not None and movies is not None, "ratings and movies dataframe must be provided"

        filtered_ratings = ratings[ratings['rating'] >= threshold_rating]
        merged_data = pd.merge(filtered_ratings, movies,
                               on='movieId', how='inner')
        transactions = merged_data.groupby(
            'userId')['title'].apply(list).reset_index()
        transactions = transactions.set_index('userId')
        # transactions_list = transactions['title'].tolist()
        transactions["title"] = transactions["title"].agg(
            lambda x: ",".join(x))
        movie_transactions = transactions["title"].str.split(',')
        return movie_transactions

    else:
        raise Exception("Invalid dataset name")
