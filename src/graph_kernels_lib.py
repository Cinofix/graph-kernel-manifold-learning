#!/usr/bin/env python
# coding: utf-8

# **Read this article presenting a way to improve the disciminative power of graph kernels.
# 
# **Choose one graph kernel among
# 
#     Shortest-path Kernel
#     Graphlet Kernel
#     Random Walk Kernel
#     Weisfeiler-Lehman Kernel
# 
# **Choose one manifold learning technique among
# 
#     Isomap
#     Diffusion Maps
#     Laplacian Eigenmaps
#     Local Linear Embedding
# 
# **Compare the performance of an SVM trained on the given kernel, with or without the manifold learning step, on the following datasets: 

# In[1]:


from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

def fit_n_components(D, Y, manifold_learning, n_neighbors = 14, n_iteration = 20):
    max_acc = 0.0
    max_idx = 1
    clf = svm.SVC(kernel="linear", C = 1.0)
    for i in range(2,n_iteration):
        ml_prj_D = manifold_learning(n_neighbors, i).fit_transform(D)
        scores_ln = cross_val_score(clf, ml_prj_D, Y, cv = 10, n_jobs= 8)
        #print("I:"+ str(i)+ " "+str(np.mean(scores_ln)))
        if np.mean(scores_ln) > max_acc:
            max_acc = np.mean(scores_ln)
            max_idx = i
    return max_idx


# # Weisfeiler Lehman Kernel 

# In[2]:


class WeisfeilerLehmanKernel():
    n_labels = 0
    compressed_labels = {} #{ hash_key: [hash]}

    def get_nodes_degree(self, graph):
        v = graph.shape[0]
        ones = np.ones((v,1))
        return np.dot(graph, ones)
    
    def get_graphs_labels(self, graphs):
        n = len(graphs)
        graphs_labels = []
        for G in graphs:
            graphs_labels.append(self.get_nodes_degree(G))
        return graphs_labels
    
    def labels_to_feature_vectors(self, graphs_degree_labels):
        n = len(graphs_degree_labels)
        size = int(np.max(np.concatenate(graphs_degree_labels)))
        degree_component = np.zeros((n, size))
        for i in range(len(graphs_degree_labels)):
            for j in graphs_degree_labels[i]:
                degree_component[i,int(j)-1] += 1
        return degree_component

    
    def get_multiset_label(self, graph, graph_labels):
        n = graph.shape[0]
        graphs_labels = np.empty((n,), dtype = np.object)
        
        for v in range(n):
            np.insert(np.nonzero(graph[v]), 0, values = v)
            neighbors = np.insert(np.nonzero(graph[v]), 0, v)
            multiset = [graph_labels[neighbor][0] for neighbor in neighbors]
            multiset[1:] = np.sort(multiset[1:])
            graphs_labels[v] = np.array(multiset)
        return graphs_labels
    
    def get_multisets_labels(self, graphs, graphs_labels):
        n = len(graphs)
        multi_labels = np.empty((n,), dtype = np.object)
        for idx in range(n):
            multi_labels[idx] = self.get_multiset_label(graphs[idx], graphs_labels[idx])
        return multi_labels
    
    def labels_compression(self, multisets_graph_labels):
        graph_cmpr_labels = {} #{hash_key: [hash, #occurences]}

        for m_labels in multisets_graph_labels:
            str_label = str(m_labels)
            if str_label not in self.compressed_labels:
                self.compressed_labels.update({str_label: self.n_labels})
                self.n_labels += 1
            
            label_hash = self.compressed_labels[str_label]
            
            if str_label not in graph_cmpr_labels:
                graph_cmpr_labels.update({str_label: [label_hash, 1]})
            else:
                value = graph_cmpr_labels[str_label]
                value[1]+= 1;
                graph_cmpr_labels.update({str_label: value})
        return graph_cmpr_labels
    
    def relabelling_graphs(self, new_labels, graphs_labels):
        n_graphs = len(new_labels)
        for i in range(n_graphs):
            cmpr_labels = self.labels_compression(new_labels[i])
            n = new_labels[i].shape[0]
            
            for v in range(n):
                node_labels = new_labels[i][v]
                f_node_labels = cmpr_labels[str(node_labels)][0]
                graphs_labels[i][v] = f_node_labels        
                
    def wl_test_graph_isomorphism(self, graphs, h):
        n = len(graphs)
        #features = np.empty((n,1), dtype = np.object)
        graphs_labels = self.get_graphs_labels(graphs)
        degree_component = self.labels_to_feature_vectors(graphs_labels)
        for i in range(h):
            self.compressed_labels = {}
            self.n_labels = 0
            new_labels = self.get_multisets_labels(graphs, graphs_labels)
            self.relabelling_graphs(new_labels, graphs_labels)
            #print("h: ",str(i))
        return np.array(graphs_labels), degree_component
    
    def extract_features_vectors(self, wl):
        n = wl.shape[0]
        features = np.zeros((n, self.n_labels))
        for i in range(n):
            for label in wl[i]:
                features[i,int(label)]+= 1
        return features
    
    def normalize(self,X):
        norms = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
        XX = X / norms
        return XX
    
    def eval_similarities(self, graphs, h):
        self.compressed_labels = {}
        self.n_labels = 0
        WL, degree_component = self.wl_test_graph_isomorphism(graphs, h)
        X = self.extract_features_vectors(WL)
        X = np.concatenate((degree_component, X), axis = 1)
        XX = self.normalize(X)
        return np.dot(XX, XX.T)


# # Shortest Path Kernel

# In[3]:


from threading import Thread
class KernelExecutor(Thread):
    
    def __init__(self, kernel, K, graphs, i):
        Thread.__init__(self)
        self.kernel_similarity = kernel
        self.k_matrix = K
        self.am_graphs = graphs
        self.idx = i
        self.v = len(self.am_graphs)
        
    def run(self):
        for j in range(self.v):
            self.k_matrix[self.idx,j] = self.kernel_similarity(self.am_graphs[self.idx], self.am_graphs[j])     
            
class FWExecutor(Thread):
    
    def __init__(self, graph_, shortest_path_method_, out_, idx_):
        Thread.__init__(self)
        self.graph = graph_
        self.shortest_path_method = shortest_path_method_
        self.out = out_
        self.idx = idx_
        
    def run(self):
        self.out[self.idx] = self.shortest_path_method(self.graph) 


# In[4]:


import numpy as np
from numpy.linalg import norm
class ShortestPathKernel():

    def initialize_paths(self, G):
        INF_ = float('inf')
        v = G.shape[0]
        dist = G
        dist[dist == 0] = INF_
        np.fill_diagonal(dist, 0)
        return dist
    
    def compute_FW_full(self, G):
        G = G.astype(np.float)
        dist = self.initialize_paths(G)
        v = G.shape[0]
        for k in range(v):
            for i in range(v):
                for j in range(v):
                    dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
        return dist
    
    def compute_FW(self, G):
        G = G.astype(np.float)
        dist = self.initialize_paths(G)
        v = G.shape[0]
        h = int(v/2)
        for k in range(v):
            for i in range(v):
                for j in range(i,v):
                    dist[i,j]= dist[j,i] = min(dist[i,j], dist[i,k] + dist[k,j])
        return dist
    
    def compute_shortest_paths(self, graphs):
        SP = []
        i = 0
        for adj_m in graphs:
            SP.append(self.compute_FW(adj_m))
            i+= 1
        return SP

    def compute_multi_shortest_paths(self, graphs):
        v = len(graphs)
        SP = np.empty((v,), dtype = np.object)
        THREAD_FOR_TIME = 6
  
        for i in range(0,v,THREAD_FOR_TIME):
            thr = []
            NTHREAD = np.minimum(v-i,THREAD_FOR_TIME)
            for j in range(NTHREAD):
                ex = FWExecutor(graphs[i+j],self.compute_FW, SP, i+j)
                thr.append(ex)
                ex.start()
            for j in range(NTHREAD):
                thr[j].join()
        return SP
    
    def extract_freq_vector(self, S, delta):
        F = np.empty([delta+1, 1])
        for i in range(delta+1):
            F[i] = np.sum(S == i)
        return F/norm(F)
    
    # similarity between frequency of paths
    def k_delta(self, SP1, SP2):
        delta = int(np.maximum(np.max(SP1), np.max(SP2)))

        F1 = self.extract_freq_vector(SP1, delta)
        F2 = self.extract_freq_vector(SP2, delta)
        return  np.dot(np.transpose(F1), F2)[0]#, F1, F2
    
    # similarity between paths weights
    def k_path_weigth(self, SP1, SP2):
        v1 = SP1.shape[0]
        v2 = SP2.shape[0]
        max_size = np.maximum(v1,v2)+1
        
        S1_l = np.sum(SP1, axis = 1)
        S2_l = np.sum(SP2, axis = 1)
        
        WS1_rows = np.concatenate([S1_l, np.zeros(max_size - v1)])# pad with zeros
        WS2_rows = np.concatenate([S2_l, np.zeros(max_size - v2)]) # pad with zeros
        return np.dot(WS1_rows, np.transpose(WS2_rows))/(norm(WS1_rows)*norm(WS2_rows))
            
    def kernel_similarity(self, SP1, SP2):
        return self.k_delta(SP1, SP2)
    
    def eval_similarities(self, SP_graphs):
        n = len(SP_graphs)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.kernel_similarity(SP_graphs[i], SP_graphs[j])
        return K

    def threads_eval_similarities(self, graphs):
        n = len(graphs)
        K = np.zeros((n, n))
        THREAD_FOR_TIME = 2
        for i in range(0, n, THREAD_FOR_TIME):
            thr = []
            for j in range(THREAD_FOR_TIME):
                ex = KernelExecutor(self.kernel_similarity, K, graphs, i+j)
                thr.append(ex)
                ex.start()
                
            for j in range(THREAD_FOR_TIME):
                thr[j].join()
        return K
