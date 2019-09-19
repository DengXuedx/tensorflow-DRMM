from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph


WALK_LEN= 50
N_WALKS= 30

def load_data(prefix, normalize=True, load_walks=False):
    train_G = nx.read_gpickle(prefix+"/graph/grid_graph.gpickle")
    conversion = lambda n : n

    if os.path.exists(prefix + "/node_features/geohash_features.npy"):
        feats = np.load(prefix + "/node_features/geohash_features.npy")#Node Feature
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    try:
        id_map = json.load(open(prefix + "/dict/geohash2id.json"))
        id_map = {conversion(k):int(v) for k,v in id_map.items()}
    except:
        id_map = dict()
        i = 0
        for n in train_G.nodes():
            id_map[n] = i
            i +=1
        with open (prefix+"/dict/geohash2id.json","w") as fr:
            fr.write(json.dumps(id_map))
    walks = []
        
   
    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in train_G.nodes() ])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))
  

    return train_G,feats, id_map,walks

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(list(G.neighbors(curr_node)))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes()]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
