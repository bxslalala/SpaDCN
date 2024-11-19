import pandas as pd
import numpy as np
import torch,ot,random,os
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


def spatial_reconstruction(
        adata: sc.AnnData,
        alpha: float = 1.0,
        n_neighbors: int = 10,
        n_pcs: int = 15,
        use_highly_variable: Optional[bool] = None,
        normalize_total: bool = False,
        copy: bool = False,
) -> Optional[sc.AnnData]:

    adata = adata.copy() if copy else adata

    adata.layers['counts'] = adata.X

    sc.pp.normalize_total(adata) if normalize_total else None
    sc.pp.log1p(adata)

    adata.layers['log1p'] = adata.X

    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_highly_variable)

    coord = adata.obsm['spatial']
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(coord)
    nbrs = neigh.kneighbors_graph(coord)
    dists = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1
    conns = nbrs.T.toarray() * dists

    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec = alpha * np.matmul(conns / np.sum(conns, axis=0, keepdims=True), X) + X
    
    conns = conns / np.sum(conns, axis=0, keepdims=True)
    
    adata.X = csr_matrix(X_rec)

    del adata.obsm['X_pca']

    adata.uns['spatial_reconstruction'] = {}

    rec_dict = adata.uns['spatial_reconstruction']

    rec_dict['params'] = {}
    rec_dict['params']['alpha'] = alpha
    rec_dict['params']['n_neighbors'] = n_neighbors
    rec_dict['params']['n_pcs'] = n_pcs
    rec_dict['params']['use_highly_variable'] = use_highly_variable
    rec_dict['params']['normalize_total'] = normalize_total

    return adata, conns


def mclust_R(adata, num_cluster, pca_num=25, modelNames='EEE', random_seed=200):
    # PCA transformation
    pca = PCA(n_components=pca_num, random_state=42)
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding

    # Setup R environment
    import rpy2.robjects as robjects
    robjects.r.options(encoding='UTF-8')
    robjects.r('Sys.setlocale("LC_ALL", "en_US.UTF-8")')

    # Suppress R console output (warnings, messages, and progress)
    robjects.r('suppressMessages(suppressWarnings(library(mclust)))')

    # Enable automatic conversion between numpy arrays and R objects
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    
    # Set random seed in R
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)

    # Convert 'embedding' to R matrix
    embedding_r = rpy2.robjects.numpy2ri.numpy2rpy(embedding)

    # Use contextlib to suppress stdout output in Jupyter Notebook
    with contextlib.redirect_stdout(io.StringIO()):
        rmclust = robjects.r['Mclust']
        fit_result = rmclust(embedding_r, G=num_cluster, modelNames=modelNames)

    # Extract the Mclust result from the R list
    mclust_res = np.array(fit_result.rx2('classification'))
    mclust_res = mclust_res.astype('int')

    # Return Mclust clustering results
    return mclust_res


def refine_label(adata, radius=50, key='pred'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    adata.obs['label_refined'] = np.array(new_type)
    
    return adata


def seed_torch(seed):
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
    # adj 是邻接矩阵 Av
    N = adj.size(0)
    I = torch.eye(N, device=adj.device)
    adj_with_self_loops = adj + I

    # 计算度矩阵 Dv
    degree_matrix = torch.diag(adj_with_self_loops.sum(dim=1))

    # 计算 Dv^(-1/2)
    D_inv_sqrt = degree_matrix.pow(-0.5)
    D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
    
     # 计算规范化邻接矩阵
    adj_normalized = torch.mm(torch.mm(D_inv_sqrt, adj_with_self_loops), D_inv_sqrt)
    
    return adj_normalized


def load_data(conns_load, features_tensor):
    
    conns = conns_load
    
    adj = conns_load
    
    edge_adj, edge_name = create_edge_adj_new(conns)
    Tmat = create_transition_matrix_new(conns)

    features = features_tensor
    edge_features = node_corr_cosine(conns, features)
    
    features = torch.FloatTensor(np.array(features))

    edge_features = torch.FloatTensor(edge_features)
    edge_feature_dict = {}
    for i in range(len(edge_name)):
        edge_feature_dict[edge_name[i]] = edge_features[i, :]
        
    Tmat = create_transition_matrix_new(conns)
    
    Tmat = sparse_mx_to_torch_sparse_tensor(Tmat)
    edge_adj = sparse_mx_to_torch_sparse_tensor(edge_adj)
    
    adj = torch.Tensor(adj)
    adj_normalized = normalize_adjacency_matrix(adj)
    return Tmat, edge_adj, adj_normalized, features, edge_features

def loss_function(p, q):
    def kld(target, pred):
        return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
    loss = kld(p, q)
    return loss
    
def target_distribution(q):
    #weight = q ** 2 / q.sum(0)
    #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
    p = q**2 / torch.sum(q, dim=0)
    p = p / torch.sum(p, dim=1, keepdim=True)
    return p



def refine_label(adata, radius=50, key='pred'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    adata.obs['label_refined'] = np.array(new_type)
    
    return adata,np.array(new_type)

def cal_ari(z,adata, domain, n_clusters, res=1):

    embedding = z.cpu().detach().numpy()
    adata_t=sc.AnnData(embedding)
        # matrix = adata.X 
        # if isinstance(matrix,sparse.spmatrix):
        #     matrix = matrix.toarray()
    adata_t.obsm['emb'] = adata_t.X

    if domain == 'louvain':
        print("Initializing cluster centers with louvain, resolution = ", res)
        sc.pp.neighbors(adata_t, n_neighbors=10)
        sc.tl.louvain(adata_t,resolution=res)
        y_pred=adata_t.obs['louvain'].astype(int).to_numpy()
        
        louvain_n_clusters=len(np.unique(y_pred))
        print("class:",louvain_n_clusters)
    elif domain == 'kmeans':
        print("Initializing cluster centers with kmeans, n_clusters known")
        kmeans = KMeans(n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(embedding)        

    elif domain =='mclust':
        print("mclust")
        ari_max = 0
        max_pca_num = 0
        for i in range(6,30):
            
            pca_num = i
            sc.pp.neighbors(adata_t, n_neighbors = 10)
            # sc.tl.umap(adata)
            y_pred = mclust_R(adata_t, num_cluster = n_clusters, pca_num = pca_num)
            adata.obs["pred"] = y_pred
            adata.obs["pred"] = adata.obs["pred"].astype('category')

            obs_df = adata.obs.dropna()
            ARI = adjusted_rand_score(obs_df['pred'], obs_df['truth'])
            
            mclust_n_clusters=len(np.unique(y_pred))

            adata,refine = refine_label(adata)
            ARI_refine = adjusted_rand_score(adata.obs['label_refined'], adata.obs['truth'])

            if(ARI > ari_max):
                ari_max = ARI
                max_pca_num = i
            if(ARI_refine > ari_max):
                ari_max = ARI_refine
                max_pca_num = i
        pca_num = max_pca_num
        sc.pp.neighbors(adata_t, n_neighbors = 10)
        y_pred = mclust_R(adata_t, num_cluster = n_clusters, pca_num = pca_num)
        adata.obs["pred"] = y_pred
        adata.obs["pred"] = adata.obs["pred"].astype('category')

        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['pred'], obs_df['truth'])
        
        mclust_n_clusters=len(np.unique(y_pred))

        print("pca_num: ", pca_num, "class :", mclust_n_clusters)
        print('Adjusted rand index = %.3f' %ARI)
        adata,refine = refine_label(adata)
        ARI_refine = adjusted_rand_score(adata.obs['label_refined'], adata.obs['truth'])
        print('refine Adjusted rand index = %.3f' %ARI_refine)
        
        
    elif domain =='leiden':
        print("Initializing cluster centers with leiden, resolution = ", res)
        sc.pp.neighbors(adata_t, n_neighbors=10)
        sc.tl.leiden(adata_t,resolution=res)
        y_pred=adata_t.obs['leiden'].astype(int).to_numpy()
        
        leiden_n_clusters=len(np.unique(y_pred))
        print("class:",leiden_n_clusters)

    if domain != 'mclust':
        adata.obs["pred"] = y_pred
        adata.obs["pred"] = adata.obs["pred"].astype('category')

        # adata = refine_label(adata, radius=radius,key='pred')

        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['pred'], obs_df['truth'])
       
        print("-----------------------------------------")
        print("pca_num: ", pca_num)
        print('Adjusted rand index = %.3f' %ARI)
    
    return adata, ARI, ARI_refine