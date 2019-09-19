from __future__ import division
from __future__ import print_function

import numpy as np
import queue 
import random

np.random.seed(123)

class EdgeMinibatchIterator(object):
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """
    def __init__(self, G, id2idx, 
            placeholders, context_pairs=None, batch_size=100, max_degree=25,
            n2v_retrain=False, fixed_n2v=False,random_walk = False, num_walks = 50, walk_len = 30,
            **kwargs):    
        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.random_walk = random_walk
        self.num_walks = num_walks
        self.walk_len = walk_len 

        self.nodes = np.random.permutation(G.nodes())
        self.adj, self.deg = self.construct_adj()
        
        if context_pairs is None:
            edges = G.edges()
        else:
            edges = context_pairs

        self.train_edges = self.edges = np.random.permutation(edges)
        self.val_edges = self.train_edges
        self.train_queue = queue.Queue()

        self.MAX_SIZE =  max(self.walk_len * self.num_walks, self.batch_size)



    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
           
            neighbors = np.array([self.id2idx[str(neighbor)] 
                for neighbor in self.G.neighbors(nodeid)
                ])
            deg[self.id2idx[str(nodeid)]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)#从邻居中随机选择max_degree个
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def end(self):
        if self.random_walk :
            return self.batch_num >=len(self.nodes)
        else:
            return self.batch_num * self.batch_size >= len(self.train_edges)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0
    
    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[str(node1)])
            batch2.append(self.id2idx[str(node2)])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def next_random_walk_feed_dict(self):
        while self.train_queue.qsize() < self.MAX_SIZE and self.batch_num < len(self.nodes):
            batch_nodes = [self.nodes[self.batch_num]]
            self.run_random_walks(batch_nodes, num_walks= self.num_walks, walk_len= self.walk_len)
            self.batch_num +=1
        
        batch_edges = []
        for i in range( self.batch_size):
            batch_edges.append( self.train_queue.get())
    
        return self.batch_feed_dict(batch_edges)


    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size, 
            len(edge_list))]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(self.val_edges), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(node_list), val_edges
        
    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)


    def run_random_walks(self, nodes, num_walks=50, walk_len = 30):
        for count, node in enumerate(nodes):
            if self.G.degree(node) == 0:
                continue
            for i in range(num_walks):
                curr_node = node
                for j in range(walk_len):
                    next_node = random.choice(list(self.G.neighbors(curr_node)))
                    # self co-occurrences are useless
                    if curr_node != node:
                        self.train_queue.put((node,curr_node))
                    curr_node = next_node
        