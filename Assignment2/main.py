import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='Heart-counts.csv',
                        help='data path')

    a = parser.parse_args()
    return(a.n_clusters, a.data)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    n_classifiers, data_path = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    X = PCA(heart.X, 100)
    # Your code
    
    # Task 2:
    ran_scores = []
    for k in range(2, 10):
        print("value of k",k)
        ran_km = KMeans(n_clusters=k, init='random')
        labels = ran_km.fit(X)
        score = ran_km.silhouette(labels, X)
        ran_scores.append(score)
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(2, 10), ran_scores)
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Random Init Kmeans Silhouette Score vs Different k value')
    plt.savefig('Task2.png')
    print(ran_scores)
    # Task 3:
    kmpp_scores = []
    for k in range(2, 10):
        kmpp = KMeans(n_clusters=k, init='kmeans++')
        labels = kmpp.fit(X)
        score = kmpp.silhouette(labels, X)
        kmpp_scores.append(score)

    fig, ax = plt.subplots()
    ax.plot(np.arange(2, 10), kmpp_scores)
    ax.set_xlabel('k')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Kmeans++ Silhouette Score vs Different k value')
    plt.savefig('Task3.png')
    print(kmpp_scores)
   
    # Task 4:
    n_classifiers = np.argmax(ran_scores) + 2
    X_2 = PCA(heart.X, 2)
    model = KMeans(n_clusters=n_classifiers, init='random')
    labels = model.fit(X_2)
    visualize_cluster(X_2, labels, labels)

def visualize_cluster(x, y, clustering):
    #Your code
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = np.unique(y)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b']
    colors = colors[:len(targets)]
    for target, color in zip(targets,colors):
        row_index = y == target
        ax.scatter(x[row_index, 0]
                , x[row_index, 1]
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig('Task4.png')


if __name__ == '__main__':
    main()