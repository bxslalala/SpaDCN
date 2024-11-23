import pandas as pd
import numpy as np
import torch, ot, random, os
import bisect
import scanpy as sc
from typing import Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse import issparse, csr_matrix
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from sklearn.decomposition import PCA
import contextlib
import io
from sklearn.metrics.cluster import adjusted_rand_score
from utils import create_edge_adj_new, create_transition_matrix_new, node_corr_cosine, create_direction_feature, normalize, sparse_mx_to_torch_sparse_tensor


# Optimized version of spatial_reconstruction function
def spatial_reconstruction(
        adata: sc.AnnData,
        alpha: float = 1.0,
        n_neighbors: int = 10,
        n_pcs: int = 15,
        use_highly_variable: Optional[bool] = None,
        normalize_total: bool = False,
        copy: bool = False,
) -> Optional[sc.AnnData]:
    """
    Performs spatial reconstruction using a combination of PCA and k-nearest neighbors for an AnnData object.

    Parameters:
    - adata: Input AnnData object.
    - alpha: Weighting parameter for reconstruction.
    - n_neighbors: Number of neighbors to consider for nearest neighbor graph.
    - n_pcs: Number of principal components to use.
    - use_highly_variable: Whether to use highly variable genes.
    - normalize_total: Whether to normalize total counts per cell.
    - copy: If True, return a copy of the AnnData object.

    Returns:
    - If copy is True, returns the reconstructed AnnData object, otherwise modifies adata in place.
    - Returns connection weights.
    """
    adata = adata.copy() if copy else adata

    # Store original counts in a separate layer
    adata.layers['counts'] = adata.X

    # Normalize and log transform the data if requested
    if normalize_total:
        sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    adata.layers['log1p'] = adata.X

    # PCA for dimensionality reduction
    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_highly_variable)

    # Construct neighbors graph based on spatial coordinates
    coord = adata.obsm['spatial']
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(coord)
    nbrs = neigh.kneighbors_graph(coord)

    # Calculate connection weights based on cosine distances in PCA space
    dists = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1
    conns = nbrs.multiply(dists).toarray()

    # Avoid dividing by zero when normalizing connections
    conns_sum = np.sum(conns, axis=0, keepdims=True)
    conns_sum[conns_sum == 0] = 1
    conns_normalized = conns / conns_sum

    # Reconstruct spatial values with the connection weights
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec = alpha * np.matmul(conns_normalized, X) + X

    # Update the AnnData with reconstructed values
    adata.X = csr_matrix(X_rec)

    # Cleanup
    del adata.obsm['X_pca']

    # Store reconstruction parameters
    adata.uns['spatial_reconstruction'] = {
        'params': {
            'alpha': alpha,
            'n_neighbors': n_neighbors,
            'n_pcs': n_pcs,
            'use_highly_variable': use_highly_variable,
            'normalize_total': normalize_total
        }
    }

    return adata, conns_normalized


def mclust_R(adata, num_cluster, pca_num=25, modelNames='EEE', random_seed=200):
    """
    Perform clustering using Mclust from R.

    Parameters:
    - adata: Input AnnData object.
    - num_cluster: Number of clusters.
    - pca_num: Number of principal components to use.
    - modelNames: Mclust model type.
    - random_seed: Random seed for reproducibility.

    Returns:
    - Clustering results as a numpy array.
    """
    # PCA transformation
    pca = PCA(n_components=pca_num, random_state=42)
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding

    # Setup R environment
    robjects.r.options(encoding='UTF-8')
    robjects.r('Sys.setlocale("LC_ALL", "en_US.UTF-8")')

    # Suppress R console output
    robjects.r('suppressMessages(suppressWarnings(library(mclust)))')

    # Enable automatic conversion between numpy arrays and R objects
    rpy2.robjects.numpy2ri.activate()

    # Set random seed in R
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)

    # Convert 'embedding' to R matrix
    embedding_r = rpy2.robjects.numpy2ri.numpy2rpy(embedding)

    # Use contextlib to suppress stdout output
    with contextlib.redirect_stdout(io.StringIO()):
        rmclust = robjects.r['Mclust']
        fit_result = rmclust(embedding_r, G=num_cluster, modelNames=modelNames)

    # Extract the Mclust result from the R list
    mclust_res = np.array(fit_result.rx2('classification')).astype('int')

    return mclust_res


def refine_label(adata, radius=50, key='pred'):
    """
    Refine cell labels by considering the labels of neighboring cells.

    Parameters:
    - adata: Input AnnData object.
    - radius: Number of neighbors to consider.
    - key: Key in adata.obs to refine.

    Returns:
    - Updated AnnData object with refined labels.
    """
    n_neigh = radius
    old_type = adata.obs[key].values

    # Calculate distance between cells
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
    n_cell = distance.shape[0]

    new_type = []
    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = [old_type[index[j]] for j in range(1, n_neigh + 1)]
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    adata.obs['label_refined'] = np.array(new_type, dtype=str)

    return adata


def seed_torch(seed):
    """
    Set random seed for reproducibility.

    Parameters:
    - seed: Random seed value.
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def normalize_adjacency_matrix(adj):
    """
    Normalize an adjacency matrix with self-loops.

    Parameters:
    - adj: Input adjacency matrix.

    Returns:
    - Normalized adjacency matrix.
    """
    N = adj.size(0)
    I = torch.eye(N, device=adj.device)
    adj_with_self_loops = adj + I

    # Calculate degree matrix
    degree_matrix = torch.diag(adj_with_self_loops.sum(dim=1))

    # Calculate D^(-1/2)
    D_inv_sqrt = degree_matrix.pow(-0.5)
    D_inv_sqrt[D_inv_sqrt == float('inf')] = 0

    # Normalize adjacency matrix
    adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj_with_self_loops), D_inv_sqrt)

    return adj_normalized


def load_data(conns_load, commot_matrix, features_tensor):
    """
    Load and preprocess data for model input.

    Parameters:
    - conns_load: Connection weights.
    - features_tensor: Feature tensor.

    Returns:
    - Transition matrix, edge adjacency, normalized adjacency, features, edge features.
    """
    commot_result = np.where(conns_load != 0, commot_matrix, 0)

    edge_adj, edge_name = create_edge_adj_new(commot_result)
    Tmat = create_transition_matrix_new(commot_result)

    features = torch.FloatTensor(np.array(features_tensor))
    edge_features = torch.FloatTensor(node_corr_cosine(commot_result, features))

    edge_feature_dict = {edge_name[i]: edge_features[i, :] for i in range(len(edge_name))}

    Tmat = sparse_mx_to_torch_sparse_tensor(Tmat)
    edge_adj = sparse_mx_to_torch_sparse_tensor(edge_adj)

    adj = torch.Tensor(conns_load)
    adj_normalized = normalize_adjacency_matrix(adj)

    return Tmat, edge_adj, adj_normalized, features, edge_features


def loss_function(p, q):
    """
    Calculate Kullback-Leibler divergence loss.

    Parameters:
    - p: Target distribution.
    - q: Predicted distribution.

    Returns:
    - KL divergence loss.
    """
    def kld(target, pred):
        return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

    return kld(p, q)


def target_distribution(q):
    """
    Compute target distribution for clustering.

    Parameters:
    - q: Soft cluster assignments.

    Returns:
    - Target distribution.
    """
    p = q**2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p


def cal_ari(z, adata, domain, n_clusters, res=1):
    """
    Calculate Adjusted Rand Index (ARI) for clustering results.

    Parameters:
    - z: Embedding tensor.
    - adata: Input AnnData object.
    - domain: Clustering method ('louvain', 'kmeans', 'mclust', 'leiden').
    - n_clusters: Number of clusters.
    - res: Resolution parameter for louvain/leiden.

    Returns:
    - Updated AnnData object, ARI, refined ARI.
    """
    embedding = z.cpu().detach().numpy()
    adata_t = sc.AnnData(embedding)
    adata_t.obsm['emb'] = adata_t.X

    if domain == 'louvain':
        print("Initializing cluster centers with louvain, resolution = ", res)
        sc.pp.neighbors(adata_t, n_neighbors=10)
        sc.tl.louvain(adata_t, resolution=res)
        y_pred = adata_t.obs['louvain'].astype(int).to_numpy()
        louvain_n_clusters = len(np.unique(y_pred))
        print("class:", louvain_n_clusters)

    elif domain == 'kmeans':
        print("Initializing cluster centers with kmeans, n_clusters known")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(embedding)

    elif domain == 'mclust':
        print("mclust")
        ari_max = 0
        max_pca_num = 0
        for i in range(6, 30):
            pca_num = i
            sc.pp.neighbors(adata_t, n_neighbors=10)
            y_pred = mclust_R(adata_t, num_cluster=n_clusters, pca_num=pca_num)
            adata.obs["pred"] = y_pred.astype('category')

            obs_df = adata.obs.dropna()
            ARI = adjusted_rand_score(obs_df['pred'], obs_df['truth'])

            adata, refine = refine_label(adata)
            ARI_refine = adjusted_rand_score(adata.obs['label_refined'], adata.obs['truth'])

            if ARI > ari_max:
                ari_max = ARI
                max_pca_num = i
            if ARI_refine > ari_max:
                ari_max = ARI_refine
                max_pca_num = i

        pca_num = max_pca_num
        sc.pp.neighbors(adata_t, n_neighbors=10)
        y_pred = mclust_R(adata_t, num_cluster=n_clusters, pca_num=pca_num)
        adata.obs["pred"] = y_pred.astype('category')

        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['pred'], obs_df['truth'])
        mclust_n_clusters = len(np.unique(y_pred))

        print("pca_num: ", pca_num, "class :", mclust_n_clusters)
        print('Adjusted rand index = %.3f' % ARI)
        adata, refine = refine_label(adata)
        ARI_refine = adjusted_rand_score(adata.obs['label_refined'], adata.obs['truth'])
        print('refine Adjusted rand index = %.3f' % ARI_refine)

    elif domain == 'leiden':
        print("Initializing cluster centers with leiden, resolution = ", res)
        sc.pp.neighbors(adata_t, n_neighbors=10)
        sc.tl.leiden(adata_t, resolution=res)
        y_pred = adata_t.obs['leiden'].astype(int).to_numpy()
        leiden_n_clusters = len(np.unique(y_pred))
        print("class:", leiden_n_clusters)

    if domain != 'mclust':
        adata.obs["pred"] = y_pred.astype('category')
        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['pred'], obs_df['truth'])
        print("-----------------------------------------")
        print("Adjusted rand index = %.3f" % ARI)

    return adata, ARI, ARI_refine
