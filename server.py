import torch
import numpy as np
import random
import networkx as nx
from dtaidistance import dtw
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster


class Server():
    def __init__(self, model, device):
        self.model = model.to(device)
        self.W = {key: value for key, value in self.model.named_parameters()}
        self.model_cache = []

    def randomSample_clients(self, all_clients, frac):
        return random.sample(all_clients, int(len(all_clients) * frac))

    def aggregate_weights(self, selected_clients):
        # pass train_size, and weighted aggregate
        total_size = 0
        for client in selected_clients:
            total_size += client.train_size
        for k in self.W.keys():
            self.W[k].data = torch.div(torch.sum(torch.stack([torch.mul(client.W[k].data, client.train_size) for client in selected_clients]), dim=0), total_size).clone()

    def compute_pairwise_similarities(self, clients):
        client_dWs = []
        for client in clients:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            client_dWs.append(dW)
        return pairwise_angles(client_dWs)
    
    def hierarchical_clustering_clients(self,clients,dist_threshold):    
        cosine_similarity = self.compute_pairwise_similarities(clients)   
        Z = linkage(cosine_similarity,method="average")
        # dendrogram(Z,above_threshold_color="#fff917")
        t1=dist_threshold
        t2=0.8
        t3=1
        # Z: distance_matrix
        # t: distance_threshold 
        clusters1 = fcluster(Z, t1, criterion='distance')
        clusters2 = fcluster(Z, t2, criterion='distance')
        clusters3 = fcluster(Z, t3, criterion='distance')
        cluster_indices = []
        cluster_indices2 = []
        cluster_indices3 = []
        print("\nclusters1:\n",clusters1)
        # print("\nclusters2:\n",clusters2)
        # print("\nclusters3:\n",clusters3)
        for x in range(np.max(clusters1)):
            cluster_indices.append(np.where(clusters1==(x+1))[0])#把类别一样的client的idc聚在一起
        # print("\ncluster idcs:\n",cluster_indices)
        # for x in range(np.max(clusters2)):
        #     cluster_indices2.append(np.where(clusters2==(x+1))[0])#把类别一样的client的idc聚在一起
        # print("\ncluster idcs:\n",cluster_indices2)
        # for x in range(np.max(clusters3)):
        #     cluster_indices3.append(np.where(clusters3==(x+1))[0])#把类别一样的client的idc聚在一起
        # print("\ncluster idcs:\n",cluster_indices3)
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
        for client_cluster in client_clusters:
            for client in client_cluster:
                print(client.name)
            print("&&&&")
        return client_clusters

    def compute_pairwise_distances(self, seqs, standardize=False):
        """ computes DTW distances """
        if standardize:
            # standardize to only focus on the trends
            seqs = np.array(seqs)
            seqs = seqs / seqs.std(axis=1).reshape(-1, 1)
            distances = dtw.distance_matrix(seqs)
        else:
            distances = dtw.distance_matrix(seqs)
        return distances

    def min_cut(self, similarity, idc):
        g = nx.Graph()
        for i in range(len(similarity)):
            for j in range(len(similarity)):
                g.add_edge(i, j, weight=similarity[i][j])
        cut, partition = nx.stoer_wagner(g)
        c1 = np.array([idc[x] for x in partition[0]])
        c2 = np.array([idc[x] for x in partition[1]])
        return c1, c2

    def aggregate_clusterwise(self, client_clusters):
        # print(client_clusters)
        for cluster in client_clusters:
            targs = []
            sours = []
            total_size = 0
            for client in cluster:
                # print("\nclient at cluster",client,"@@@@",cluster)
                # if len(client)==0:
                #     break
                W = {}
                dW = {}
                for k in self.W.keys():
                    W[k] = client.W[k]#这里的client.W是更新之前的
                    dW[k] = client.dW[k]
                targs.append(W)
                sours.append((dW, client.train_size))
                total_size += client.train_size
            # pass train_size, and weighted aggregate
            reduce_add_average(targets=targs, sources=sours, total_size=total_size)

    def compute_max_update_norm(self, cluster):
        max_dW = -np.inf
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            update_norm = torch.norm(flatten(dW)).item()
            if update_norm > max_dW:
                max_dW = update_norm
        return max_dW
        # return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        cluster_dWs = []
        for client in cluster:
            dW = {}
            for k in self.W.keys():
                dW[k] = client.dW[k]
            cluster_dWs.append(flatten(dW))

        return torch.norm(torch.mean(torch.stack(cluster_dWs), dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs,
                              {name: params[name].data.clone() for name in params},
                              [accuracies[i] for i in idcs])]

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1)
            s2 = flatten(source2)
            angles[i, j] = torch.true_divide(torch.sum(s1 * s2), max(torch.norm(s1) * torch.norm(s2), 1e-12)) + 1

    return angles.numpy()

def reduce_add_average(targets, sources, total_size):
    for target in targets:
        for name in target:
            tmp = torch.div(torch.sum(torch.stack([torch.mul(source[0][name].data, source[1]) for source in sources]), dim=0), total_size).clone()
            target[name].data += tmp