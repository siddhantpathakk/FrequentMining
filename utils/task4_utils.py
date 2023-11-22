import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import interpolate
from scipy.spatial import ConvexHull

class VotingClassifier:
    def __init__(self, estimators, type, estimator_names):
        self.estimators = estimators
        self.type = type
        self.estimator_names = estimator_names

    def fit_predict(self, df):
        individual_predictions = {}
        for c, estimator in enumerate(self.estimators):
            clusters = estimator.fit_predict(df)
            individual_predictions[self.estimator_names[c]] = clusters

        return individual_predictions

    def voting(self, individual_predictions, num_classes, num_samples):
        if self.type == 'hard':
            return self.hard_voting(individual_predictions, num_classes, num_samples)
        # elif self.type == 'soft':
        #     return self.soft_voting(individual_predictions, num_classes, num_samples)
        else:
            raise ValueError('type must be either hard or soft')

    def hard_voting(self, individual_predictions, num_classes, num_samples):
        # simply count the votes of each estimator class
        # and return the most voted class
        new_votes = []
        for i in range(num_samples):
            cur_votes = self.get_cur_votes(individual_predictions, i)
            most_voted = np.argmax(np.bincount(cur_votes))
            new_votes.append(most_voted)

        return new_votes

    def get_cur_votes(self, individual_predictions, idx):
        cur_val = []
        for estimator_class in self.estimator_names:
            cur_val.append(individual_predictions[estimator_class][idx])
        return cur_val


def plot_clusters(estimator_names, ip, votes, df_new, target):

    # Reduce dimensionality for visualization (adjust n_components as needed)
    pca = PCA(n_components=2)
    trans_data_pca = pca.fit_transform(df_new)

    num_est = len(estimator_names)
    fig, ax = plt.subplots(1, num_est+2)
    
    fig.set_size_inches(10*(num_est+2), 8)
    
    for i,est in enumerate(estimator_names):
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