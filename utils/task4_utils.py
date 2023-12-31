import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from scipy import interpolate
# from scipy.spatial import ConvexHull

from utils.task3_utils import *

def _init_empty_dict(minsup_vals, algos):
    final_dict = {}
    for algo in algos:
        final_dict[algo] = {}
        for minsup in minsup_vals:
            final_dict[algo][minsup] = []
    return final_dict

def _fix_dictionary(_ans_dict):
    new_dict = {}
    for algo in _ans_dict:
        new_dict[algo] = {}
        for minsup in _ans_dict[algo]:
            if not isinstance(_ans_dict[algo][minsup], list):
                new_dict[algo][minsup] = _ans_dict[algo][minsup]
    return new_dict
                

def tune_minsup(df, target, num_classes, minsup_values, algorithms):
    """
    Tune minsup for each algorithm and return the best minsup for all algorithms

    Args:
        df (pd.DataFrame): Dataframe of the dataset
        target (pd.Series): Target column of the dataset
        num_classes (int): Number of classes in the dataset
        minsup_values (list): List of minsup values to try
        algorithms (list): List of algorithms to try

    Returns:
        float: Best minsup value for all algorithms
    """
    best_minsups = []
    accs_dict = _init_empty_dict(minsup_values, algorithms)
    
    cutoff = 1
    for minsup in minsup_values:
        
        if minsup > cutoff:
            break
        
        frequent_itemsets = get_frequent_itemsets(df, minsup)
        
        if len(frequent_itemsets) == 0:
            cutoff = minsup
            break
        
        for algo in algorithms:            
            best_accuracy_for_algo = 0
            best_minsup_for_algo = None

            df_transformed = make_cluster_df(df, frequent_itemsets)
            clusters = do_clustering(df_transformed, algo, n_clusters=num_classes)
            accuracy, _ = remap_labels(clusters, target)
            accs_dict[algo][minsup] = accuracy

            if accuracy > best_accuracy_for_algo:
                best_accuracy_for_algo = accuracy
                best_minsup_for_algo = minsup
                # print(f'Best minsup for {algo}:\t{best_minsup_for_algo:.3f}\tAccuracy:\t{best_accuracy_for_algo:.4f}')
                best_minsups.append(best_minsup_for_algo)

    accs_dict = _fix_dictionary(accs_dict)

    # get minsup with highest accuracy for each algorithm
    best_minsups_dict = {}
    for algo in algorithms:
        best_minsups_dict[algo] = max(accs_dict[algo], key=accs_dict[algo].get)

    best_minsup_for_all = max(best_minsups_dict, key=best_minsups_dict.get)
    best_minsup_val = best_minsups_dict[best_minsup_for_all]
    
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16, 8)
    
    for algo in accs_dict:
        ax.plot(accs_dict[algo].keys(), accs_dict[algo].values(), label=algo)
        
    for algo in accs_dict:
        ax.scatter(best_minsups_dict[algo], accs_dict[algo][best_minsups_dict[algo]], c='r', marker = 'x', s=100)
        
    ax.set_xlabel('Minsup')
    ax.set_ylabel('Accuracy')
    ax.set_title('Minsup vs Accuracy')
    ax.legend()    
    plt.show()


    return round(best_minsup_val,3)


class VotingClassifier:
    def __init__(self, estimators, type, estimator_names):
        self.estimators = estimators
        self.type = type
        self.estimator_names = estimator_names

    def fit_predict(self, df, target):
        """
        Fit the estimators on the dataset and return the predictions of each estimator

        Args:
            df (pd.DataFrame): Dataframe of the dataset
            target (pd.Series): Target column of the dataset

        Returns:
            dict: Dictionary of predictions of each estimator
        """
        individual_predictions = {}
        for c, estimator in enumerate(self.estimators):
            clusters = estimator.fit_predict(df)
            _, remapped_labels = remap_labels(clusters, target)
            individual_predictions[self.estimator_names[c]] = remapped_labels
            
        return individual_predictions

    def voting(self, individual_predictions, num_classes, num_samples, target, weights_of_estimators=None):
        """
        Perform voting on the predictions of each estimator

        Args:
            individual_predictions (dict): Dictionary of predictions of each estimator
            num_classes (int): Number of classes in the dataset
            num_samples (int): Number of samples in the dataset
            target (pd.Series): Target column of the dataset
            weights_of_estimators (list, optional): List of weights for each estimator. Defaults to None.

        Raises:
            NotImplementedError: Only hard voting is implemented for now. Please use type="hard"

        Returns:
            list: List of votes for each sample
        """
        if self.type == 'hard':
            votes = self.hard_voting(individual_predictions, num_samples)
            acc, votes = remap_labels(votes, target)
            print(f'Accuracy for voting classifier:\t{acc:.4f}\n')
            return votes
        else:
            raise NotImplementedError('Only hard voting is implemented for now. Please use type="hard"')

    def hard_voting(self, individual_predictions, num_samples):
        """
        Perform hard voting on the predictions of each estimator

        Args:
            individual_predictions (dict): Dictionary of predictions of each estimator
            num_samples (int): Number of samples in the dataset

        Returns:
            list: List of votes for each sample
        """
        new_votes = []
        for i in range(num_samples):
            cur_votes = self.get_cur_votes(individual_predictions, i)
            most_voted = np.argmax(np.bincount(cur_votes))
            new_votes.append(most_voted)
        return new_votes
    
    def get_cur_votes(self, individual_predictions, idx):
        """
        Get the votes for a particular sample

        Args:
            individual_predictions (dict): Dictionary of predictions of each estimator
            idx (int): Index of the sample

        Returns:
            list: List of votes for the sample
        """
        cur_val = []
        for estimator_class in self.estimator_names:
            cur_val.append(individual_predictions[estimator_class][idx])
        return cur_val


def plot_clusters(estimator_names, ip, votes, df_new, target, plot_all=False):
    """
    Plot the clusters of each estimator and the voting

    Args:
        estimator_names (list): List of names of the estimators
        ip (dict): Dictionary of predictions of each estimator
        votes (list): List of votes for each sample
        df_new (pd.DataFrame): Dataframe of the dataset
        target (pd.Series): Target column of the dataset
        plot_all (bool, optional): Whether to plot all the estimators or just the voting. Defaults to False.
    """

    # Reduce dimensionality for visualization (adjust n_components as needed)
    pca = PCA(n_components=2)
    trans_data_pca = pca.fit_transform(df_new)

    if plot_all:
        num_est = len(estimator_names)
        fig, ax = plt.subplots(1, num_est+2)
        
        fig.set_size_inches(10*(num_est+2), 8)
        
        for i, est in enumerate(estimator_names):
            scatter = ax[i].scatter(trans_data_pca[:, 0],  trans_data_pca[:, 1], c=ip[est], cmap='Set1', alpha=0.7)
            
            # # draw the convex hull around the clusters for each estimator
            # for i in np.unique(target):
            #     points = trans_data_pca[ip[est] == i]
            #     hull = ConvexHull(points)
                
            #     x_hull = np.append(points[hull.vertices,0],
            #                points[hull.vertices,0][0])
            #     y_hull = np.append(points[hull.vertices,1],
            #                     points[hull.vertices,1][0])
                
            #     plt.fill(x_hull, y_hull, alpha=0.3, c='red')
                    
            
            legend = ax[i].legend(*scatter.legend_elements(),
                                loc="lower left", title=est)
            
            ax[i].add_artist(legend)
        
        scatter_2 = ax[num_est].scatter(trans_data_pca[:, 0], trans_data_pca[:, 1], c=target, cmap='Set1', alpha=0.7)
        legend_2 = ax[num_est].legend(*scatter_2.legend_elements(), loc="lower left", title="Actual Dataset")
        ax[num_est].add_artist(legend_2)
        
        scatter_1 = ax[num_est+1].scatter(trans_data_pca[:, 0], trans_data_pca[:, 1], c=votes, cmap='Set1', alpha=0.7)
        legend_1 = ax[num_est+1].legend(*scatter_1.legend_elements(), loc="lower left", title="Voting")
        ax[num_est+1].add_artist(legend_1)
        
        plt.show()
    
    else:
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(20, 8)
        
        scatter_2 = ax[0].scatter(trans_data_pca[:, 0], trans_data_pca[:, 1], c=target, cmap='Set1', alpha=0.7)
        legend_2 = ax[0].legend(*scatter_2.legend_elements(), loc="lower left", title="Actual Dataset")
        ax[0].add_artist(legend_2)
        
        scatter_1 = ax[1].scatter(trans_data_pca[:, 0], trans_data_pca[:, 1], c=votes, cmap='Set1', alpha=0.7)
        legend_1 = ax[1].legend(*scatter_1.legend_elements(), loc="lower left", title="Voting")
        ax[1].add_artist(legend_1)
        
        plt.show()
        
        plot_clusters_3d(df_new, votes, target)
         # plot_target_3d(df_new, target)
    
    
def plot_clusters_3d(df_new, clusters, target):
    pca = PCA(n_components=3)
    trans_data_pca = pca.fit_transform(df_new)
    fig = plt.figure(figsize=(24, 8))
    
    ax = fig.add_subplot(121, projection='3d')
    scatter = ax.scatter(trans_data_pca[:, 0], trans_data_pca[:, 1],
                         trans_data_pca[:, 2], c=clusters, cmap='viridis', s=50)
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    legend1 = ax.legend(*scatter.legend_elements(), loc="best", title="Classes")
    ax.add_artist(legend1)
    plt.title('Predicted clusters')
    
    
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(trans_data_pca[:, 0], trans_data_pca[:, 1],
                         trans_data_pca[:, 2], c=target, cmap='viridis', s=50)
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.set_zlabel('Principal Component 3')
    
    legend2 = ax2.legend(*scatter2.legend_elements(), loc="best", title="Classes")
    ax2.add_artist(legend2)
    plt.title('Actual Clusters')

    plt.show()