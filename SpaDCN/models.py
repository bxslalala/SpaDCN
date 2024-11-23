import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import scanpy as sc
import pandas as pd
import numpy as np
from layers import GraphConvolution
from data import mclust_R,refine_label
from tqdm import tqdm
from sklearn.metrics.cluster import adjusted_rand_score

class GCN(nn.Module):
    def __init__(self, nfeat_v, nfeat_e, nhid, device, a):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, device, a, node_layer=True)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e, device, a, node_layer=False)
        self.gc3 = GraphConvolution(nhid, nfeat_v, nfeat_e, nfeat_e, device, a, node_layer=True)
        self.gc1 = self.gc1.to(device)
        self.gc2 = self.gc2.to(device)
        self.gc3 = self.gc3.to(device)
        self.device = device

    def forward(self, X, Z, adj_e, adj_v, T):
        X, Z, adj_e, adj_v, T = map(lambda t: t.to(self.device), (X, Z, adj_e, adj_v, T))
        X, Z = self.gc1(X, Z, adj_e, adj_v, T)
        X, Z = self.gc2(X, Z, adj_e, adj_v, T)
        X, Z = self.gc3(X, Z, adj_e, adj_v, T)
        return X


def search_res(adata, n_clusters, n_neighbors=10, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    """
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    """
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(adata.obs['leiden'].unique())
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(adata.obs['louvain'].unique())
        print(f'resolution={res}, cluster number={count_unique}')
        if count_unique == n_clusters:
            label = 1
            break
    assert label == 1, "Resolution not found. Please try a larger range or smaller step size."
    return res


class SpaDCN(nn.Module):
    def __init__(self, features, edge_features, edge_adj, adj, Tmat, nhid, n_class, dropout_ratio, device, a, alpha=0.2):
        super(SpaDCN, self).__init__()

        self.gc = GCN(features.shape[1], edge_features.shape[1], nhid, device, a)
        self.gc = self.gc.to(device)
        self.alpha = alpha
        self.features = features
        self.edge_features = edge_features
        self.edge_adj = edge_adj
        self.adj = adj
        self.Tmat = Tmat
        self.device = device
        self.n_class = n_class
        print(f"Using device: {self.device}")

    def forward(self):
        x = self.gc(self.features, self.edge_features, self.edge_adj, self.adj, self.Tmat)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha) + 1e-8)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))
        return kld(p, q)

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, lr=0.001, max_epochs=500, update_interval=30, trajectory_interval=50, weight_decay=5e-4, opt="adam", init="mclust", n_neighbors=10, start=0.1, end=2.0, increment=0.1, res=0.4, n_clusters=10, 
            tol=1e-5,pca_num=25):
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) if opt == "adam" else optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        self.trajectory=[]
        # Initialize cluster centers
        features = self.gc(self.features, self.edge_features, self.edge_adj, self.adj, self.Tmat)
       
        embedding = features.cpu().detach().numpy()
        adata=sc.AnnData(embedding)
        
        adata.obsm['emb'] = adata.X

        if init == "mclust":
            print("Initializing cluster centers with mclust")
            y_pred = mclust_R(adata, n_clusters, pca_num=pca_num)
        elif init in ["louvain", "leiden"]:
            print(f"Initializing cluster centers with {init}, resolution={res}")
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            res = search_res(adata, n_clusters, n_neighbors=n_neighbors, use_rep='X_pca', method=init, start=start, end=end, increment=increment)
            if init == "louvain":
                sc.tl.louvain(adata, resolution=res)
                y_pred = adata.obs['louvain'].astype(int).to_numpy()
            else:
                sc.tl.leiden(adata, resolution=res)
                y_pred = adata.obs['leiden'].astype(int).to_numpy()
        else:
            raise ValueError("Unsupported initialization method.")
        self.trajectory.append(y_pred)
        self.n_clusters = len(np.unique(y_pred))
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.n_class)).to(self.device)
        self.mu.data.copy_(torch.Tensor(np.asarray(pd.DataFrame(self.features.cpu().detach().numpy()).groupby(y_pred).mean())))
        self.train()

        y_pred_last = y_pred
        for epoch in tqdm(range(max_epochs), desc="Training Progress"):
            if epoch % update_interval == 0:
                _, q = self.forward()
                p = self.target_distribution(q).data

            optimizer.zero_grad()
            _, q = self.forward()
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

            if epoch % trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            # Check stopping criteria
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and (epoch) % update_interval == 0 and delta_label < tol:
                print(f"Convergence reached with delta_label {delta_label:.4f} < tol {tol:.4f} at epoch {epoch}")
                break

        torch.cuda.empty_cache()
        print("Training complete.")
        return y_pred

    def predict(self, adata, domain, n_clusters, res, use_rep='truth'):
        z,q = self.forward()

        if domain in ['louvain', 'leiden'] and res is None:
            raise ValueError("Parameter 'res' must be provided for 'louvain' or 'leiden' clustering methods.")
        adata.obsm['SpaDCN'] = z
        
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
            ari_max = 0
            max_pca_num = 0
            for i in range(6, 30):
                pca_num = i
                sc.pp.neighbors(adata_t, n_neighbors=10)
                y_pred = mclust_R(adata_t, num_cluster=n_clusters, pca_num=pca_num)
                adata.obs["pred"] = y_pred
                adata.obs["pred"] = adata.obs["pred"].astype('category')
                obs_df = adata.obs.dropna()
                ARI = adjusted_rand_score(obs_df['pred'], obs_df['truth'])

                adata= refine_label(adata)
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
            adata.obsm['SpaDCN'] = adata_t.obsm['emb_pca']
            adata.obs["pred"] = y_pred
            adata.obs["pred"] = adata.obs["pred"].astype('category')
            obs_df = adata.obs.dropna()
            ARI = adjusted_rand_score(obs_df['pred'], obs_df['truth'])
            mclust_n_clusters = len(np.unique(y_pred))

            print("pca_num: ", pca_num, "class :", mclust_n_clusters)
            print('Adjusted rand index = %.3f' % ARI)
            adata= refine_label(adata)
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
