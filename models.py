import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import scanpy as sc
import pandas as pd
import numpy as np
from layers import GraphConvolution
from data import mclust_R
from tqdm import tqdm

class GCN(nn.Module):
    def __init__(self, nfeat_v, nfeat_e, nhid, device, a, node_layer=True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, device, a, node_layer=True)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e,device,a, node_layer=False)
        self.gc3 = GraphConvolution(nhid, nfeat_v, nfeat_e, nfeat_e,device,a, node_layer=True)
        self.gc1 = self.gc1.to(device)
        self.gc2 = self.gc2.to(device)
        self.gc3 = self.gc3.to(device)
        self.device = device


    def forward(self, X, Z, adj_e, adj_v, T):

        X = X.to(self.device)
        Z = Z.to(self.device)
        adj_e = adj_e.to(self.device)
        adj_v = adj_v.to(self.device)
        T = T.to(self.device)
        gc1 = self.gc1(X, Z, adj_e, adj_v, T)
        
        X , Z = gc1[0], gc1[1]

        gc2 = self.gc2(X, Z, adj_e, adj_v, T)
        X , Z = gc2[0], gc2[1]
        
        X, Z = self.gc3(X, Z, adj_e, adj_v, T)

        return X


def search_res(adata, n_clusters, n_neighbors=10,method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
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
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res    


class GEN(nn.Module):
    def __init__(self, features, edge_features, edge_adj, adj, Tmat, nhid, n_class, dropout_ratio, device, a,alpha=0.2):
        super(GEN, self).__init__()
    
        self.gc = GCN(features.shape[1], edge_features.shape[1], nhid, device, a)
        self.gc = self.gc.to(device)
        self.nhid=nhid
        self.n_class=n_class
        #self.mu determined by the init method
        self.alpha=alpha
        self.features=features
        self.edge_features=edge_features
        self.edge_adj=edge_adj
        self.adj=adj
        self.Tmat=Tmat
        self.device=device
        print("self.device",self.device)

    def forward(self):
        x=self.gc(self.features, self.edge_features, self.edge_adj, self.adj, self.Tmat)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1).to(self.device) - self.mu)**2, dim=2).to(self.device) / self.alpha) + 1e-8)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        #weight = q ** 2 / q.sum(0)
        #return torch.transpose((torch.transpose(weight,0,1) / weight.sum(1)),0,1)e
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X, edge_features, edge_adj, adj, Tmat, pca_num=25, lr=0.001, max_epochs=500, update_interval=30, 
            trajectory_interval=50,weight_decay=5e-4,opt="admin",init="mclust",n_neighbors=10,start=0.1,end=2.0,increment=0.1,res=0.4,n_clusters=10,init_spa=True,tol=1e-3):
        self.trajectory=[]
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)

       
        features = self.gc(X, edge_features, edge_adj, adj, Tmat)
        # features = X
        
        # pca = PCA(n_components=pca_num, random_state=42) 
        # embedding = pca.fit_transform(features.cpu().detach().numpy())
        
        # return features
        embedding = features.cpu().detach().numpy()
        

        adata=sc.AnnData(embedding)
        
        adata.obsm['emb'] = adata.X
        
        #----------------------------------------------------------------        
        if init=="kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters=n_clusters
            kmeans = KMeans(self.n_clusters, n_init=20)
            if init_spa:
                #------Kmeans use exp and spatial
                y_pred = kmeans.fit_predict(features.cpu().detach().numpy())
            else:
                #------Kmeans only use exp info, no spatial
                y_pred = kmeans.fit_predict(X)  #Here we use X as numpy
        elif init=="louvain":
            print("Initializing cluster centers with louvain, resolution = ", res)
            if init_spa:
                adata=sc.AnnData(features.cpu().detach().numpy())
            else:
                adata=sc.AnnData(X)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            
            res = search_res(adata, n_clusters, n_neighbors=n_neighbors, use_rep='emb_pca', method=init, start=start, end=end, increment=increment)
            sc.tl.louvain(adata,resolution=res)
            y_pred=adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters=len(np.unique(y_pred))
        elif init=="mclust":
            print("Initializing cluster centers with mclust, n_clusters known")
            sc.pp.neighbors(adata, n_neighbors)
            # sc.tl.umap(adata)
            self.n_clusters=n_clusters
            y_pred = mclust_R(adata, self.n_clusters, pca_num)
            # y_pred=adata.obs['mclust'].astype(int).to_numpy()
        elif init=="leiden":
            print("Initializing cluster centers with leiden, resolution = ", res)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors)
            res = search_res(adata, n_clusters, n_neighbors=n_neighbors, use_rep='emb_pca', method=init, start=start, end=end, increment=increment)
            sc.tl.leiden(adata,resolution=res)
            y_pred=adata.obs['leiden'].astype(int).to_numpy()
            self.n_clusters=len(np.unique(y_pred))
        #----------------------------------------------------------------
        y_pred_last = y_pred
        yyy = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.n_class)).to(self.device)
        # self.mu = Parameter(torch.Tensor(self.n_clusters, 128))
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        self.trajectory.append(y_pred)
        features=pd.DataFrame(features.cpu().detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(y_pred,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        
        
        # self.mu = Parameter(torch.Tensor(5, self.n_class))
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        
        
        
        


        for epoch in tqdm(range(max_epochs), desc="Training Progress"):
                    
            # print("self.mu:", self.mu)
            if epoch % update_interval == 0:
                _, q = self.forward()
                p = self.target_distribution(q).data
            optimizer.zero_grad()
            z, q = self.forward()
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()
            
            # print("epoch", epoch, "loss:", loss)
            
            if epoch % trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            # Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and (epoch - 1) % update_interval == 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                print("Total epoch:", epoch)
                break
                
        torch.cuda.empty_cache()
        print("Train over.")
        return yyy

        
        
        
        
        # for epoch in range(max_epochs):
            
        #     # print("self.mu:", self.mu)
        #     if epoch%update_interval == 0:
        #         _, q = self.forward()
        #         p = self.target_distribution(q).data
        #     if epoch%10==0:
        #         print("Epoch ", epoch) 
        #     optimizer.zero_grad()
        #     z,q = self.forward()
        #     loss = self.loss_function(p, q)
        #     loss.backward()
        #     optimizer.step()
            
        #     # print("epoch", epoch, "loss:", loss)
            
        #     if epoch%trajectory_interval == 0:
        #         self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

        #     #Check stop criterion
        #     y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        #     delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
        #     y_pred_last = y_pred
        #     if epoch>0 and (epoch-1)%update_interval == 0 and delta_label < tol:
        #         print('delta_label ', delta_label, '< tol ', tol)
        #         print("Reach tolerance threshold. Stopping training.")
        #         print("Total epoch:", epoch)
        #         break
        # torch.cuda.empty_cache()
        # print("train over.")
        # return yyy

    def fit_with_init(self, X,adj, init_y, lr=0.001, max_epochs=5000, update_interval=1, weight_decay=5e-4,opt="sgd"):
        print("Initializing cluster centers with kmeans.")
        if opt=="sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt=="admin":
            optimizer = optim.Adam(self.parameters(),lr=lr, weight_decay=weight_decay)
        X=torch.FloatTensor(X)
        adj=torch.FloatTensor(adj)
        features, _ = self.forward(X,adj)
        features=pd.DataFrame(features.detach().numpy(),index=np.arange(0,features.shape[0]))
        Group=pd.Series(init_y,index=np.arange(0,features.shape[0]),name="Group")
        Mergefeature=pd.concat([features,Group],axis=1)
        cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        for epoch in range(max_epochs):
            if epoch%update_interval == 0:
                _, q = self.forward(torch.FloatTensor(X),torch.FloatTensor(adj))
                p = self.target_distribution(q).data
            X=torch.FloatTensor(X)
            adj=torch.FloatTensor(adj)
            optimizer.zero_grad()
            z,q = self(X, adj)
            loss = self.loss_function(p, q)
            loss.backward()
            optimizer.step()

    def predict(self):
        z,q = self.forward()
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        # Max probability plot
        prob=q.cpu().detach().numpy()
        return y_pred, prob, z, q